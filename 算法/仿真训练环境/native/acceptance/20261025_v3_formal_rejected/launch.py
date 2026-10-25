from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, shlex, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE=Path(__file__).resolve().parent
REPO=Path('/mnt/c/pioneer/先进通用履带底盘工程')
NATIVE=REPO/'算法'/'仿真训练环境'/'native'
TRAIN=NATIVE/'train'
TOOLS=NATIVE/'tools'
TAG='fixed_physics_v3_20261025_184052'
DISTRO='Ubuntu-24.04-Arm64'
WINDOWS_LAUNCHER=BASE/'windows_launcher.json'
sys.path.insert(0,str(TOOLS))
import benchmark_training as bt

def now_pair():
    u=datetime.now(timezone.utc).replace(microsecond=0)
    return {'utc':u.isoformat().replace('+00:00','Z'),'beijing':u.astimezone(timezone(timedelta(hours=8))).isoformat()}
def sha_bytes(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write_json(path:Path,obj):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def versions():
    v=bt.versions()
    for name in ('gymnasium','scipy','psutil','packaging','tensorboard'):
        try:v[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:v[name]=None
    return v
def fail(reason,manifest=None):
    stamp=now_pair()
    if manifest is None:manifest={'schema':'fixed_physics_v3_launch_v1'}
    manifest.update(status='aborted_before_launch',error=reason,completed_at_utc=stamp['utc'],completed_at_beijing=stamp['beijing'])
    write_json(BASE/'launch_manifest.json',manifest)
    print(json.dumps({'status':'aborted_before_launch','error':reason},ensure_ascii=False),file=sys.stderr)
    return 2
def metrics_state(run:Path|None):
    if run is None:return {'run_dir':None,'metrics_rows_complete':0,'iteration':None,'contract_sha256':None,'nonfinite_events_bytes':0,'failure_json_present':False}
    p=run/'metrics.jsonl';rows=[];parse_error=None
    if p.is_file():
        raw=p.read_bytes();parts=raw.splitlines(keepends=True)
        for line in parts:
            if not line.endswith(b'\n'):continue
            try:rows.append(json.loads(line.decode('utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))))
            except Exception as exc:parse_error=f'{type(exc).__name__}: {exc}'
    contract_sha=None
    cp=run/'observation_contract.json'
    if cp.is_file():
        try:contract_sha=json.loads(cp.read_text(encoding='utf-8')).get('observation_contract_sha256')
        except Exception:pass
    nf=run/'nonfinite_events.jsonl'
    return {'run_dir':str(run),'metrics_rows_complete':len(rows),'iteration':rows[-1].get('iter') if rows else None,
            'last_nonfinite_env_resets':rows[-1].get('nonfinite_env_resets') if rows else None,
            'last_nonfinite_actions':rows[-1].get('nonfinite_actions') if rows else None,
            'last_wall_s':rows[-1].get('wall_s') if rows else None,'last_samples_per_s':rows[-1].get('samples_per_s') if rows else None,
            'contract_sha256':contract_sha,'metrics_parse_error':parse_error,
            'nonfinite_events_bytes':nf.stat().st_size if nf.is_file() else 0,'failure_json_present':(run/'failure.json').is_file()}
def find_run():
    matches=sorted((TRAIN/'runs').glob(f'native_v1_1_{TAG}_*')) if (TRAIN/'runs').is_dir() else []
    return matches[-1] if matches else None
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-release-notified-at',required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    prep=json.loads((BASE/'prepare_manifest.json').read_text(encoding='utf-8'))
    source_identity=json.loads((BASE/'source_identity.json').read_text(encoding='utf-8'))
    manifest={'schema':'fixed_physics_v3_launch_v1','status':'preparing','tag':TAG,'seed':0,'resume':False,
        'resume_checkpoint':None,'root_release_notified_at':args.root_release_notified_at,
        'controller_started_at':now_pair(),'controller_pid_wsl':os.getpid(),'launcher_pid_windows':None,
        'wrapper_pid_wsl':None,'source_identity_prepared':source_identity,'dependency_versions_prepared':prep['dependency_versions'],
        'capability_claim':False,'capability_metrics_allowed':False,'restart_policy':'never restart or change iteration budget',
        'rawlog_paths':{'wrapper_stdout':'wrapper.stdout.rawlog','wrapper_stderr':'wrapper.stderr.rawlog',
            'gpu_samples':'nvidia_smi_samples.jsonl','monitor':'monitor.jsonl'},'monitor_interval_s':30}
    # Recheck source bytes and package identity directly before the only GPU window.
    current=bt.source_identity();current_versions=versions()
    if current['git']['relevant_sources_dirty'] is not False:return fail('relevant_sources_dirty is not false',manifest)
    if current['git']['commit']!=source_identity['git']['commit'] or current['git']['branch']!=source_identity['git']['branch'] or current['manifest_sha256']!=source_identity['manifest_sha256']:
        return fail('source identity differs from prepared raw snapshot',manifest)
    if current_versions!=prep['dependency_versions']:return fail('dependency versions differ from prepared manifest',manifest)
    manifest['source_identity_at_launch']=current;manifest['dependency_versions_at_launch']=current_versions
    # Verify the separately snapshotted wrappers and both frozen protocol SHA files.
    control_results=[]
    for item in prep['control_file_snapshot']['files']:
        rel=Path(item['path']);src=TRAIN/rel.name;snap=BASE/'control_files'/rel
        match=src.is_file() and snap.is_file() and sha_file(src)==item['source_sha256']==sha_file(snap)
        control_results.append({'path':item['path'],'sha256':item['source_sha256'],'match':match})
    manifest['control_files_at_launch']=control_results
    if not all(x['match'] for x in control_results):return fail('wrapper/protocol control-file bytes changed',manifest)
    # CPU-only full path/snapshot/process gate; this branch must not query nvidia-smi or start training.
    required_paths=[BASE,REPO,NATIVE,TRAIN,TOOLS,TRAIN/'run_native_v1_trial.sh',TRAIN/'run_native_v1_trial.ps1',
        TRAIN/'train_wsl.py',TRAIN/'trial_checks.py',TRAIN/'PROTOCOL_NATIVE_V1.1.sha256',TOOLS/'benchmark_training.py',
        BASE/'launch.py',BASE/'prepare_manifest.json',BASE/'source_identity.json',BASE/'launch_windows.ps1']
    path_rows=[];path_errors=[]
    for path in required_paths:
        try:
            resolved=path.resolve(strict=True);path_rows.append({'path':str(path),'resolved':str(resolved),'exists':True})
        except OSError as exc:
            path_rows.append({'path':str(path),'resolved':None,'exists':False});path_errors.append(f'missing path: {path}: {exc}')
    source_rows=[];source_errors=[]
    repo_root=Path(prep['repo_root']).resolve(strict=True)
    for row in source_identity['raw_source_bytes']:
        rel=Path(row['path']);src=(repo_root/rel);snap=BASE/'source_snapshot'/rel
        try:
            src_resolved=src.resolve(strict=True);snap_resolved=snap.resolve(strict=True)
            src_hash=sha_file(src_resolved);snap_hash=sha_file(snap_resolved)
            ok=src_hash==row['sha256']==snap_hash
            source_rows.append({'path':row['path'],'source_resolved':str(src_resolved),'snapshot_resolved':str(snap_resolved),
                'source_sha256':src_hash,'snapshot_sha256':snap_hash,'match':ok})
            if not ok:source_errors.append(f'source snapshot hash mismatch: {row["path"]}')
        except OSError as exc:
            source_rows.append({'path':row['path'],'match':False,'error':f'{type(exc).__name__}: {exc}'})
            source_errors.append(f'missing source/snapshot path: {row["path"]}')
    scan=bt.process_scan();busy=[x for x in scan if set(x.get('labels',[])) & {'training','llama','kit','isaac'}]
    run_prefix=f'native_v1_1_{TAG}_';existing=list((TRAIN/'runs').glob(run_prefix+'*')) if (TRAIN/'runs').is_dir() else []
    cpu_errors=path_errors+source_errors
    if current['git']['relevant_sources_dirty'] is not False:cpu_errors.append('relevant_sources_dirty is not false')
    if current['manifest_sha256']!=source_identity['manifest_sha256']:cpu_errors.append('source identity manifest changed')
    if current_versions!=prep['dependency_versions']:cpu_errors.append('dependency version set changed')
    if not all(x['match'] for x in control_results):cpu_errors.append('wrapper/protocol control files do not match')
    if busy:cpu_errors.append(f'known WSL training/inference/Kit process detected: {busy}')
    if existing:cpu_errors.append(f'tag run directory already exists: {[str(x) for x in existing]}')
    cpu_preflight={'schema':'fixed_physics_v3_cpu_preflight_v1','created_at':now_pair(),'status':'pass' if not cpu_errors else 'fail',
        'errors':cpu_errors,'gpu_query_count':0,'training_started':False,'windows_process_pid_observed':True,
        'windows_launcher_pid_from_record':json.loads(WINDOWS_LAUNCHER.read_text(encoding='utf-8')).get('windows_pid') if WINDOWS_LAUNCHER.is_file() else None,
        'source_identity':current,'dependency_versions':current_versions,'path_checks':path_rows,
        'source_snapshot_checks':source_rows,'control_file_checks':control_results,'wsl_process_scan':scan,
        'run_tag':TAG,'existing_tag_runs':[str(x) for x in existing]}
    if args.preflight_only:
        write_json(BASE/'launch_preflight_cpu.json',cpu_preflight)
        print(json.dumps({'status':cpu_preflight['status'],'errors':cpu_errors,'source_files':len(source_rows),
            'control_files':len(control_results),'processes':len(scan),'gpu_query_count':0},ensure_ascii=False,sort_keys=True))
        return 0 if not cpu_errors else 2
    prefix=f'native_v1_1_{TAG}_'
    existing=list((TRAIN/'runs').glob(prefix+'*')) if (TRAIN/'runs').exists() else []
    if existing:return fail(f'tag already has run directory: {[str(x) for x in existing]}',manifest)
    # The Windows parent records its PID only after Start-Process; wait before GPU preflight.
    deadline=time.monotonic()+30
    while not WINDOWS_LAUNCHER.is_file() and time.monotonic()<deadline:time.sleep(0.25)
    if not WINDOWS_LAUNCHER.is_file():return fail('Windows launcher PID metadata not received',manifest)
    winrec=json.loads(WINDOWS_LAUNCHER.read_text(encoding='utf-8'))
    manifest['launcher_pid_windows']=winrec.get('windows_pid');manifest['windows_launcher_record']=winrec
    before=bt.gpu_snapshot();manifest['gpu_before']=before
    busy=[x for x in before.get('processes',[]) if set(x.get('labels',[])) & {'training','llama','kit','isaac'}]
    if not before.get('verified') or before.get('compute_apps') or busy:return fail('GPU/process preflight not idle or unverifiable',manifest)
    # Final source identity check immediately before the wrapper starts.
    start_source=bt.source_identity()
    if start_source['git']['relevant_sources_dirty'] is not False or start_source['manifest_sha256']!=source_identity['manifest_sha256']:
        return fail('source changed after preflight; training not started',manifest)
    wrapper=TRAIN/'run_native_v1_trial.sh'
    child_argv=['/usr/bin/env','SIM_DT=0.001','DECIMATION=50','PYTHON=/opt/venv/bin/python','/bin/bash',str(wrapper),TAG]
    manifest['wrapper_argv']=child_argv;manifest['wrapper_cwd']=str(TRAIN);manifest['status']='starting'
    write_json(BASE/'launch_manifest.json',manifest)
    out_path=BASE/'wrapper.stdout.rawlog';err_path=BASE/'wrapper.stderr.rawlog'
    stop=threading.Event();count=[0]
    start=time.perf_counter()
    try:
        with out_path.open('wb') as outlog,err_path.open('wb') as errlog:
            child=subprocess.Popen(child_argv,cwd=str(TRAIN),stdin=subprocess.DEVNULL,stdout=outlog,stderr=errlog,shell=False,close_fds=True)
            manifest.update(status='running',started_at=now_pair(),wrapper_pid_wsl=child.pid)
            write_json(BASE/'launch_manifest.json',manifest)
            sampler=threading.Thread(target=bt._sample_gpu,args=(BASE/'nvidia_smi_samples.jsonl',stop,count),daemon=True)
            sampler.start()
            monitor_path=BASE/'monitor.jsonl'
            next_sample=time.monotonic()
            while child.poll() is None:
                current_mono=time.monotonic()
                if current_mono>=next_sample:
                    run=find_run();ms=metrics_state(run)
                    snap={'sampled_at':now_pair(),'elapsed_s':round(current_mono-start,3),'wrapper_pid_wsl':child.pid,
                        'windows_pid':manifest['launcher_pid_windows'],'run':ms,
                        'processes':bt.process_scan(),'gpu_samples_collected':count[0]}
                    with monitor_path.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(snap,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    manifest['latest_monitor_snapshot']=snap;write_json(BASE/'launch_manifest.json',manifest)
                    next_sample=current_mono+30.0
                time.sleep(0.5)
            rc=child.wait();wrapper_wall=time.perf_counter()-start
    except Exception as exc:
        stop.set()
        manifest.update(status='launcher_exception',error=f'{type(exc).__name__}: {exc}',wrapper_wall_s=time.perf_counter()-start)
        write_json(BASE/'launch_manifest.json',manifest);return 2
    stop.set();sampler.join(timeout=3)
    run=find_run();ms=metrics_state(run)
    with (BASE/'monitor.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps({'sampled_at':now_pair(),'elapsed_s':round(wrapper_wall,3),'wrapper_pid_wsl':child.pid,
            'windows_pid':manifest['launcher_pid_windows'],'run':ms,'processes':bt.process_scan(),
            'gpu_samples_collected':count[0],'wrapper_returncode':rc},ensure_ascii=False,allow_nan=False)+'\n')
    after=bt.gpu_snapshot();finish_source=bt.source_identity()
    manifest.update(status='completed' if rc==0 else 'failed',completed_at=now_pair(),wrapper_returncode=rc,
        wrapper_wall_s=wrapper_wall,trainer_summary_wall_s=None,gpu_after=after,gpu_sample_count=count[0],
        source_identity_after=finish_source,source_identity_unchanged=(finish_source['git']['commit']==start_source['git']['commit'] and finish_source['manifest_sha256']==start_source['manifest_sha256']),
        run_directory=str(run) if run else None,run_progress=ms)
    if run:
        summary_path=run/'summary.json';cli_path=run/'cli_args.txt';manifest_path=run/'trial_manifest.json'
        if summary_path.is_file():
            summary=json.loads(summary_path.read_text(encoding='utf-8'));manifest['trainer_summary_wall_s']=summary.get('total_wall_s')
        if cli_path.is_file():
            raw=cli_path.read_bytes();(BASE/'actual_cli_args.txt').write_bytes(raw)
            try:manifest['cli_args_recorded_argv']=shlex.split(raw.decode('utf-8'))
            except Exception as exc:manifest['cli_args_recorded_argv_parse_error']=f'{type(exc).__name__}: {exc}'
            manifest['actual_train_argv']=['/opt/venv/bin/python','-u',str(TRAIN/'train_wsl.py'),
                '--policy','lstm','--lstm-hidden','512','--num-envs','64','--num-steps','32',
                '--iterations','300','--seed','0','--threads','8','--device','cuda','--physics','newton_xpbd',
                '--task-backend','fast_stop_v2','--lidar','warp','--feedback','vqf','--depth','analytic',
                '--detections','analytic','--detections-warp','--sim-dt','0.001','--decimation','50',
                '--save-interval','20','--nonfinite-limit','0','--out',run.relative_to(TRAIN).as_posix()]
            manifest['cli_args_recording_order_note']='cli_args.txt is wrapper-formatted with train_wsl.py before -u; actual execution argv follows the wrapper execution line with -u before train_wsl.py.'
        copy_names=['metrics.jsonl','nonfinite_events.jsonl','train.log','identity.json','observation_contract.json','summary.json','trial_manifest.json','cli_args.txt','physics.txt','PROTOCOL_NATIVE_V1.1.md','PROTOCOL_NATIVE_V1.1.sha256','PROTOCOL_NATIVE_V1.md','PROTOCOL_NATIVE_V1.sha256','failure.json']
        copy_root=BASE/'run_snapshot';copy_root.mkdir(exist_ok=True);copy_rows=[]
        for name in copy_names:
            src=run/name
            if not src.is_file():continue
            raw=src.read_bytes();dst=copy_root/name;dst.write_bytes(raw);copy_rows.append({'name':name,'bytes':len(raw),'source_sha256':sha_bytes(raw),'copy_sha256':sha_file(dst),'byte_identical':sha_bytes(raw)==sha_file(dst)})
        write_json(BASE/'run_snapshot_manifest.json',{'source_run':str(run),'files':copy_rows,'all_byte_identical':all(x['byte_identical'] for x in copy_rows),'checkpoints_copied':False})
        if manifest_path.is_file():manifest['trial_manifest']=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['source_identity_unchanged']=manifest.get('source_identity_unchanged') is True
    manifest['capability_claim']=False;manifest['capability_metrics_allowed']=False
    write_json(BASE/'launch_manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'returncode':rc,'wrapper_wall_s':wrapper_wall,'run_directory':manifest.get('run_directory'),'contract_sha256':ms.get('contract_sha256')},ensure_ascii=False))
    return 0 if rc==0 else rc
if __name__=='__main__':raise SystemExit(main())