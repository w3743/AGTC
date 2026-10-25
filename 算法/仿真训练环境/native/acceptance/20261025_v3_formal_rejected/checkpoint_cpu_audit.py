import hashlib, json, torch
from pathlib import Path
from torch.torch_version import TorchVersion
from torch.serialization import safe_globals
src=Path('/mnt/c/pioneer/agtc-native-acceptance-backup/fixed_physics_v3_prep_20261025_184052/checkpoint_for_lfs/checkpoint_40.pt')
out=Path('/mnt/c/pioneer/agtc-native-acceptance-backup/fixed_physics_v3_prep_20261025_184052/checkpoint_cpu_audit.json')
torch.set_num_threads(1)
cuda_before=torch.cuda.is_initialized()
with safe_globals([TorchVersion]): obj=torch.load(src,map_location='cpu',weights_only=True)
cuda_after=torch.cuda.is_initialized()
def audit_tree(root):
    result={'tensor_count':0,'floating_tensor_count':0,'tensor_elements':0,'floating_elements':0,'nonfinite_tensor_count':0,'nonfinite_elements':0}
    step_values=[]
    def walk(x,key='root'):
        if torch.is_tensor(x):
            result['tensor_count']+=1; result['tensor_elements']+=x.numel()
            if x.is_floating_point() or x.is_complex():
                result['floating_tensor_count']+=1; result['floating_elements']+=x.numel()
                mask=torch.isfinite(x)
                if not bool(mask.all().item()):
                    result['nonfinite_tensor_count']+=1
                    result['nonfinite_elements']+=int((~mask).sum().item())
            if key=='step' and x.numel()==1:
                try: step_values.append(int(x.item()))
                except Exception: pass
        elif isinstance(x,dict):
            for k,v in x.items():
                if k=='step' and not torch.is_tensor(v):
                    try: step_values.append(int(v))
                    except Exception: pass
                walk(v,str(k))
        elif isinstance(x,(list,tuple)):
            for i,v in enumerate(x): walk(v,str(i))
    walk(root)
    result['optimizer_step_count']=len(step_values)
    result['optimizer_step_min']=min(step_values) if step_values else None
    result['optimizer_step_max']=max(step_values) if step_values else None
    return result
summary={
 'schema':'checkpoint_cpu_audit_v1',
 'checkpoint_path_windows':r'C:\pioneer\agtc-native-acceptance-backup\fixed_physics_v3_prep_20261025_184052\checkpoint_for_lfs\checkpoint_40.pt',
 'checkpoint_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
 'checkpoint_bytes':src.stat().st_size,
 'load_mode':'torch.load(weights_only=True, map_location=cpu; TorchVersion safe-global only)',
 'torch_version':torch.__version__,
 'cuda_initialized_before':cuda_before,
 'cuda_initialized_after':cuda_after,
 'iteration':obj.get('iteration'),
 'task_backend':obj.get('task_backend'),
 'identity_schema_version':obj.get('identity_schema_version'),
 'observation_contract_sha256':obj.get('observation_contract_sha256'),
 'policy':audit_tree(obj.get('policy')),
 'optimizer':audit_tree(obj.get('optimizer')),
 'whole_payload':audit_tree(obj),
 'interpretation':'Tensor finiteness only; does not validate physical correctness or make the failed trial acceptable.'
}
out.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'written':str(out),'iteration':summary['iteration'],'policy':summary['policy'],'optimizer':summary['optimizer'],'cuda_initialized_after':cuda_after,'checkpoint_sha256':summary['checkpoint_sha256']},sort_keys=True))