import hashlib,json,pathlib,subprocess,datetime
repo=pathlib.Path("/mnt/c/pioneer/先进通用履带底盘工程")
commit="f0e3bea7"
relative="算法/仿真训练环境/native/proto/train_m0.py"
tree=subprocess.run(["git","ls-tree",commit,"--",relative],cwd=repo,capture_output=True,check=False)
lines=tree.stdout.decode("utf-8","replace").splitlines()
if not lines: result={"path":relative,"commit":commit,"blob_found":False}
else:
 meta,path=lines[0].split("\t",1); mode,kind,oid=meta.split()
 obj=subprocess.run(["git","cat-file","blob",oid],cwd=repo,capture_output=True,check=False)
 current=(repo/relative).read_bytes()
 result={
  "path":relative,"commit":commit,"git_blob_oid":oid,"git_blob_mode":mode,
  "git_object_type":kind,"committed_blob_bytes":len(obj.stdout),
  "committed_blob_sha256":hashlib.sha256(obj.stdout).hexdigest(),
  "worktree_bytes":len(current),"worktree_sha256":hashlib.sha256(current).hexdigest(),
  "worktree_matches_committed_blob_byte_for_byte":current==obj.stdout,
  "git_commands_used":["git ls-tree","git cat-file blob"],
  "index_or_status_read":False
 }
result["addendum_created_at_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")
pathlib.Path("/mnt/c/pioneer/agtc-native-acceptance-backup/pilot06_observation_20261025_171104/source_hash_addendum.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))