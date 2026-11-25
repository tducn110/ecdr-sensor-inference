#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
from pathlib import Path
root=Path('/home/pro/Downloads/research')
folder=root/'output/week1_closeout_20261102/orchestration'
name=sys.argv[1]
prompt=(folder/(name+'.prompt.txt')).read_text()
limit=1200 if name.startswith('implementation_fix') or name in ('subagent_feedback_implement_verifier','verifier_serialization_fix','verifier_schema_fix','verifier_csv_fix_and_run') else (900 if name.startswith('implementation_code') else 300)
session_flags=['--conversation',sys.argv[2]] if len(sys.argv)>2 else ['--new-project']
cmd=['agy',*session_flags,'--model','gemini-3.8-flash-medium','--effort','medium','--output-format','stream-json','--print-timeout',str(limit)+'s','--log-file',str(folder/(name+'.agy.log')),'--print='+prompt]
meta={'worker':name,'model':'gemini-3.8-flash-medium','effort':'medium','start_unix':time.time(),'timeout_seconds':limit+60,'prompt_path':str(folder/(name+'.prompt.txt')),'argv_without_prompt':cmd[:-1]}
try:
 with (folder/(name+'.stdout.jsonl')).open('w') as out, (folder/(name+'.stderr.log')).open('w') as err:
  proc=subprocess.Popen(cmd,cwd=root,stdout=out,stderr=err,start_new_session=True)
  meta['pid']=proc.pid
  (folder/(name+'.meta.json')).write_text(json.dumps(meta,indent=2)+'\n')
  try: meta['returncode']=proc.wait(timeout=limit+60)
  except subprocess.TimeoutExpired:
   meta['timed_out']=True
   os.killpg(proc.pid,signal.SIGTERM)
   try: meta['returncode']=proc.wait(timeout=10)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGKILL);meta['returncode']=proc.wait()
except Exception as exc:
 meta['launcher_exception']=repr(exc)
finally:
 meta['end_unix']=time.time()
 (folder/(name+'.meta.json')).write_text(json.dumps(meta,indent=2)+'\n')
 registry=Path('/home/pro/.gemini/antigravity-cli/cache/last_conversations.json')
 if registry.exists():
  registry_data=json.loads(registry.read_text())
  registry_data[str(root)]='23d34d98-7ecd-486d-af9f-6bd27562ef8b'
  temporary=registry.with_name(registry.name+'.week1tmp')
  temporary.write_text(json.dumps(registry_data,indent=2)+'\n')
  temporary.replace(registry)
 print(json.dumps(meta,indent=2))
