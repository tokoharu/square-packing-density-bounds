"""Run all independently verifiable angle cases. Nonzero exit means incomplete."""
import argparse,concurrent.futures,json,subprocess,time,hashlib
from pathlib import Path

def run(r):
    out=subprocess.run(['./verify',str(r),str(r)],capture_output=True,text=True)
    if out.returncode:raise RuntimeError(f'r={r}, returncode={out.returncode}: {out.stderr}')
    d=json.loads(out.stdout);assert d['r']==r and d['status']=='verified' and d['lower_bound']>=1.0001
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    Path('verification_summary.json').unlink(missing_ok=True)
    subprocess.run(['g++','-O2','-std=c++17','-fno-fast-math','-ffp-contract=off','verify.cpp','-o','verify'],check=True)
    start=time.time();result=[]
    with Path('verified_angles.jsonl').open('w') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run,r) for r in range(201)]
        for future in concurrent.futures.as_completed(jobs):
            row=future.result();result.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if len(result)%10==0 or len(result)==201:print(json.dumps({'completed':len(result),'total':201,'elapsed_seconds':time.time()-start,'nodes':sum(x['nodes'] for x in result)}),flush=True)
    assert sorted(r['r'] for r in result)==list(range(201))
    summary={'status':'VERIFIED','angle_cases':201,'target_lower_bound_exact':'10001/10000','nodes':sum(x['nodes'] for x in result),'leaves':sum(x['leaves'] for x in result),'minimum_printed_leaf_lower_bound':min(x['lower_bound'] for x in result),'wall_seconds':time.time()-start,'input_sha256':hashlib.sha256(Path('certificate_input.txt').read_bytes()).hexdigest(),'verifier_source_sha256':hashlib.sha256(Path('verify.cpp').read_bytes()).hexdigest(),'method':'Outward-rounded binary64 intervals; certified inscribed polygon area; derivative bounds over center boxes; rational angular net.'}
    Path('verification_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
