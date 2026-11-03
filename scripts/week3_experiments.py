#!/usr/bin/env python3
"""Reproducible Week 3 actual-graph diagnostics and fail-closed complete traces.

No positive float64 IBP result is called an executable certificate. Validation
chooses a radius before test data are loaded; labels never select runtime actions.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, resource, sys, time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ecdr.data import load_full_stream
from ecdr.week3_verifier import FrozenK12Reference,ActualGraphIBP,RuntimeGate,diagonal_box,fixed_overlap_wave
from scripts.week2_model_audit import metrics


def json_value(value):
    """Convert NumPy scalars recursively; retain strict rejection of NaN/Inf."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return [json_value(v) for v in value]
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
    return value
def dump(path,data):path.write_text(json.dumps(json_value(data),indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
            'p95':float(np.quantile(a,.95)),'total':float(a.sum())} if len(a) else None
def adjacent(df):
    return np.flatnonzero((df.exp.to_numpy()[1:]==df.exp.to_numpy()[:-1]) &
                         (df.win_idx.to_numpy()[1:]==df.win_idx.to_numpy()[:-1]+1))
def normalize_batch(ref,raw):return ((raw-ref.mean[None,:,None])/ref.std[None,:,None]).astype(np.float32)
def batch_logits(model,z):
    with torch.inference_mode():return np.concatenate([model(torch.from_numpy(z[i:i+256])).numpy() for i in range(0,len(z),256)])
def stratum(row):
    if row.label_policy=='pure':return 'pure_basic' if 1<=row.label_majority<=6 else 'pure_transition'
    return {'majority':'majority_mixed','ambiguous':'ambiguous','unlabeled':'unlabeled'}[row.label_policy]


def select_policy(ref,verifier,out):
    raw,df=load_full_stream('val');z=normalize_batch(ref,raw);pairs=adjacent(df)
    sample=pairs[np.linspace(0,len(pairs)-1,min(64,len(pairs)),dtype=int)]
    radii=[0.0,1e-6,1e-4,.001,.01,.1,1.0]
    protocol={'date':'2026-11-03','scope':'exploratory feasibility, frozen before held-out test loading',
       'radius_grid_normalized':radii,'validation_anchor_indices':sample.tolist(),
       'domain_kinds':['diagonal_box','fixed_overlap_wave'],
       'wave_selector':'constant continuation [old suffix, old suffix]; first half fixed',
       'selection':'maximize sum of conditional-positive next memberships across both families; tie: conditional positives then smallest positive radius; zero radius control ineligible',
       'test_matching':'both families same chosen radius, same anchor/winner; domain ablation, different centers/free coordinates, no general superiority claim',
       'matched_center_control':'full box at wave center, same tail radius/method/winner; wave is its fixed-overlap slice, different volume',
       'runtime':'max_age128, fixed trusted anchor, build on missing certificate, all formal reuse disabled',
       'baseline_scope':'same RAM-resident raw float32 windows, CPU, 1 torch thread; includes conversion, normalization and transfer/forward; loading setup reported separately',
       'runtime_repetitions':2,'optional_pca_regions':'DEFERRED until numerical verifier valid; H10 geometry still measured'}
    dump(out/'protocol.json',protocol)
    results=[]
    for radius in radii:
        for kind in protocol['domain_kinds']:
            positive=members=joint=0;times=[]
            for i in sample:
                r=verifier.build(raw[i],radius,recording_id=str(int(df.iloc[i].exp)),step_idx=int(df.iloc[i].win_idx),domain_kind=kind)
                member=r.domain.contains(z[i+1]) if r.domain else False
                positive+=r.conditional_positive;members+=member;joint+=r.conditional_positive and member;times.append(r.total_ms)
            results.append({'radius':radius,'kind':kind,'attempts':len(sample),'conditional_positive':positive,
                            'next_members':members,'conditional_positive_members':joint,'build_ms':stats(times)})
    positive_radii=[r for r in radii if r>0]
    def rank(r):
        entries=[v for v in results if v['radius']==r]
        return (sum(v['conditional_positive_members'] for v in entries),sum(v['conditional_positive'] for v in entries),-r)
    selected=max(positive_radii,key=rank)
    protocol['selected_radius']=selected;protocol['validation_results']=results
    protocol['frozen_at_utc']=datetime.now(timezone.utc).isoformat();dump(out/'protocol.json',protocol)
    print('validation selected radius',selected,flush=True)
    return selected,protocol


def geometry(ref,out):
    raw,df=load_full_stream('train');z=normalize_batch(ref,raw);pairs=adjacent(df)
    delta=(z[pairs+1]-z[pairs]).reshape(len(pairs),-1).astype(np.float64)
    center=delta.mean(0);cov=np.cov(delta,rowvar=False)
    eigenvalues,eigenvectors=np.linalg.eigh(cov);basis=eigenvectors[:,-3:][:,::-1]
    # Geometry-only validation observations. Runner-up competitor uses frozen logits.
    rawv,dfv=load_full_stream('val');zv=normalize_batch(ref,rawv);vp=adjacent(dfv)
    sample=vp[np.linspace(0,len(vp)-1,32,dtype=int)];obs=[]
    for i in sample:
        x=torch.from_numpy(zv[i:i+1].copy()).requires_grad_(True)
        logits=ref.model(x);top=logits.detach()[0].argsort(descending=True)[:2]
        g=torch.autograd.grad(logits[0,top[0]]-logits[0,top[1]],x)[0].numpy().reshape(-1).astype(float)
        d=(zv[i+1]-zv[i]).reshape(-1).astype(float);gn=np.linalg.norm(g);dn=np.linalg.norm(d)
        obs.append({'val_index':int(i),'winner':int(top[0]),'competitor':int(top[1]),
                    'temporal_abs_cosine':float(abs(g@d)/(gn*dn)) if gn*dn else None,
                    'pca_top3_normal_projection':float(np.linalg.norm(basis.T@g)/gn) if gn else None})
    result={'status':'VERIFIED_DESCRIPTIVE_LOCAL_GEOMETRY','train_pair_count':len(pairs),'val_gradient_count':len(obs),
       'space':'normalized 768 coordinates; gradient of winner-minus-runner-up logit at validation anchor',
       'pca_fit':'training adjacent ordinary-coordinate deltas, centered; covariance eigendecomposition',
       'pca_top3_explained_variance':float(eigenvalues[-3:].sum()/eigenvalues.sum()),
       'temporal_abs_cosine':stats([r['temporal_abs_cosine'] for r in obs if r['temporal_abs_cosine'] is not None]),
       'pca_top3_normal_projection':stats([r['pca_top3_normal_projection'] for r in obs if r['pca_top3_normal_projection'] is not None]),
       'interpretation':'large normal projection means locally sensitive; small projection is tangent-like, not a whole-region certificate; no PCA superiority established',
       'observations':obs}
    dump(out/'geometry.json',result)
    return result


def conditional_domains(ref,verifier,raw,df,z,logits,radius,out):
    valid_next=set(adjacent(df).tolist());pred=logits.argmax(1);records=[];summaries={}
    for kind in ['diagonal_box','fixed_overlap_wave']:
        builds=[];bound=[];positive=member=joint=shadow_dis=0;fail=Counter();strata={}
        matched=[]
        for i,row in enumerate(df.itertuples(index=False)):
            r=verifier.build(raw[i],radius,recording_id=str(int(row.exp)),step_idx=int(row.win_idx),domain_kind=kind)
            has_next=i in valid_next
            is_member=bool(has_next and r.domain and r.domain.contains(z[i+1]))
            joint_member=is_member and r.conditional_positive
            mismatch=bool(joint_member and pred[i+1]!=r.target_class)
            positive+=r.conditional_positive;member+=is_member;joint+=joint_member;shadow_dis+=mismatch
            fail[r.reason]+=1;builds.append(r.total_ms);bound.append(r.bound_ms)
            s=stratum(df.iloc[i+1] if has_next else df.iloc[i]);ss=strata.setdefault(s,Counter())
            ss['query_denominator' if has_next else 'terminal_anchor']+=1
            ss['candidate_members']+=is_member;ss['conditional_positive_members']+=joint_member
            improve=float(np.min(np.array(r.lower_margins)-np.array(r.separate_lower_margins))) if r.lower_margins else None
            records.append({'kind':kind,'global_row_id':int(row.global_row_id),'exp':int(row.exp),'win_idx':int(row.win_idx),
                'target':r.target_class,'conditional_positive':r.conditional_positive,'formal_certified':False,
                'free_dimensions':r.domain.free_dimensions if r.domain else None,'has_next':has_next,'next_member':is_member,
                'conditional_member':joint_member,'shadow_disagreement':mismatch,'min_lower':r.min_lower_margin,
                'min_direct_improvement':improve,'reason':r.reason,'reference_ms':r.reference_ms,'bound_ms':r.bound_ms,'build_ms':r.total_ms})
            # Same-center full-box control on fixed, declared deterministic anchor subset.
            if kind=='fixed_overlap_wave' and i%40==0:
                center=np.concatenate([z[i,:,64:],z[i,:,64:]],axis=1)
                direct,separate=verifier.bounds(diagonal_box(center,radius),r.target_class)
                matched.append({'test_index':i,'wave_positive':r.conditional_positive,'full_same_center_positive':bool(np.all(direct>0)),
                    'wave_min_lower':r.min_lower_margin,'full_min_lower':float(direct.min()),
                    'same_tail_radius':radius,'wave_free':r.domain.free_dimensions,'full_free':768})
            if (i+1)%1000==0:print(kind,i+1,'/',len(df),flush=True)
        summaries[kind]={'attempted_builds':len(df),'conditional_positive_builds':positive,'validated_numerical_certificates':0,
          'p_build_conditional':positive/len(df),'p_build_formal':0.0,'candidate_next_members':member,
          'p_member_full_stream':member/len(df),'p_member_given_adjacent':member/len(valid_next),
          'conditional_positive_member_queries':joint,'formal_reused_queries':0,'r_cert':0.0,'A_reuse':None,
          'conditional_shadow_disagreements':shadow_dis,'failure_or_conditional_reasons':dict(fail),
          'build_ms':stats(builds),'bound_ms':stats(bound),'stratification':{k:dict(v) for k,v in strata.items()},
          'same_center_control':matched,'evidence_status':'CONDITIONAL_DIAGNOSTIC_ONLY'}
    with (out/'domain_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    dump(out/'domain_results.json',summaries)
    return summaries


def benchmark(ref,verifier,raw,df,shadow,radius,out):
    # Warmup excludes instrumentation from traces; RAM data residency is identical.
    for i in range(32):ref.infer(raw[i])
    traces=[];runs=[];baseline_preds=None
    for repeat in range(2):
        # Ordinary complete CNN core, without per-query identity rehashing.
        # Validation, normalization and CPU tensor conversion remain included.
        core=[];cp=[]
        for x in raw:
            t=time.perf_counter();normalized=ref.normalize(x)
            with torch.inference_mode():v=ref.model(torch.from_numpy(normalized[None])).numpy()[0]
            cp.append(int(v.argmax()));core.append((time.perf_counter()-t)*1000)
        assert np.array_equal(cp,shadow),'core batch1 differs from frozen reference decisions'
        baseline=[];bp=[]
        begin=time.perf_counter()
        for x in raw:
            t=time.perf_counter();ref.assert_identity();z=ref.normalize(x)
            with torch.inference_mode():v=ref.model(torch.from_numpy(z[None])).numpy()[0]
            bp.append(int(v.argmax()));baseline.append((time.perf_counter()-t)*1000)
        baseline_wall=(time.perf_counter()-begin)*1000
        assert np.array_equal(bp,shadow),'batch1 vs batch256 decision changed; contract must be revised'
        baseline_preds=np.array(bp)
        for kind in ['diagonal_box','fixed_overlap_wave']:
            gate=RuntimeGate(ref,verifier);lat=[];members=[];build=[];reference=[];reasons=Counter();build_reasons=Counter();n_build=conditional_positive=0;reuse=disagreement=0
            t0=time.perf_counter()
            for i,row in enumerate(df.itertuples(index=False)):
                decision=gate.predict(raw[i],recording_id=str(int(row.exp)),step_idx=int(row.win_idx),radius=radius,domain_kind=kind,max_age=128)
                lat.append(decision.total_ms);members.append(decision.membership_ms);build.append(decision.build_ms);reference.append(decision.reference_ms)
                reasons[decision.reason]+=1;reuse+=decision.is_reuse;disagreement+=decision.pred_class!=shadow[i]
                if decision.build_result:
                    n_build+=1;conditional_positive+=decision.build_result.conditional_positive;build_reasons[decision.build_result.reason]+=1
                traces.append({'repeat':repeat,'kind':kind,'global_row_id':int(row.global_row_id),'exp':int(row.exp),'win_idx':int(row.win_idx),
                  'stratum':stratum(df.iloc[i]),'pred':decision.pred_class,'reuse':decision.is_reuse,'member':decision.is_member,
                  'anchor_age':decision.certificate_age,'reason':decision.reason,'build_reason':decision.build_result.reason if decision.build_result else '',
                  'reference_ms':decision.reference_ms,'membership_ms':decision.membership_ms,'build_ms':decision.build_ms,'total_ms':decision.total_ms})
            wall=(time.perf_counter()-t0)*1000
            runs.append({'repeat':repeat,'kind':kind,'windows':len(raw),'accepted_reuse':reuse,'fallback_count':len(raw)-reuse,
                'disagreements_full_stream':disagreement,'A_reuse':None if not reuse else (reuse-disagreement)/reuse,
                'attempted_builds':n_build,'conditional_positive_builds':conditional_positive,'build_reasons':dict(build_reasons),
                'fallback_reasons':dict(reasons),'baseline_ms':stats(baseline),'reference_core_ms':stats(core),'ecdr_ms':stats(lat),
                'reference_component_ms':stats(reference),'membership_component_ms':stats(members),'build_component_ms':stats(build),
                'baseline_loop_wall_ms':baseline_wall,'ecdr_loop_wall_ms':wall,'query_trace_speedup':sum(baseline)/sum(lat),
                'core_reference_speedup':sum(core)/sum(lat),
                'saving_ms':sum(baseline)-sum(lat),
                'note':'baseline_ms is frozen-wrapper baseline including full identity rehash; reference_core_ms excludes that rehash. ECDR includes state, input validation, identity, normalize, RAM-to-tensor, fallback and attempted builds (including duplicate anchor inference on build). CSV/summary/stratification outside query timers; whole-loop walls additionally include collection overhead'})
            print('runtime',repeat,kind,'builds',n_build,'speedup',runs[-1]['query_trace_speedup'],flush=True)
    with (out/'runtime_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(traces[0]));writer.writeheader();writer.writerows(traces)
    dump(out/'runtime_results.json',{'runs':runs,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'memory_scope':'process peak includes data, PCA and verifier allocations; not incremental cache overhead or GPU memory',
       'shadow_scope':'reference logits precomputed outside timed runtime; no shadow inference inside ECDR query path',
       'cost_scope':'RAM-resident CPU replay, no disk/network in per-query path; input loading/setup reported separately; no energy measurement'})
    return runs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=ROOT/'output/week3_certification_20261103');args=ap.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    setup=time.perf_counter();ref=FrozenK12Reference.load();verifier=ActualGraphIBP(ref);setup_ms=(time.perf_counter()-setup)*1000
    radius,protocol=select_policy(ref,verifier,out);g=geometry(ref,out)
    # Test split first loaded after writing the frozen validation-selected policy.
    loading=time.perf_counter();raw,df=load_full_stream('test');df=df.reset_index(drop=True);load_ms=(time.perf_counter()-loading)*1000
    assert len(raw)==5053 and len(adjacent(df))==5034
    z=normalize_batch(ref,raw);logits=batch_logits(ref.model,z);pred=logits.argmax(1)
    saved=np.load(ref.reference_dir/'test_predictions.npz');assert np.array_equal(pred,saved['predictions'])
    sup=df.label_policy.isin(['pure','majority']).to_numpy() & df.label_majority.between(1,12).to_numpy()
    _,m=metrics(df.label_majority.to_numpy()[sup]-1,pred[sup],12);assert m['n']==3572
    domains=conditional_domains(ref,verifier,raw,df,z,logits,radius,out)
    runs=benchmark(ref,verifier,raw,df,pred,radius,out)
    results={'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
       'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'torch_threads':torch.get_num_threads()},
       'reference_identity':asdict(ref.identity),'selected_radius':radius,'full_test_windows':len(raw),'supervised_test_windows':int(sup.sum()),
       'adjacent_test_pairs':len(adjacent(df)),'test_recordings':int(df.exp.nunique()),'label_strata':dict(Counter(stratum(r) for _,r in df.iterrows())),
       'supervised_reference_metrics':m,'fresh_vs_frozen_prediction_disagreements':int((pred!=saved['predictions']).sum()),
       'setup_ms':setup_ms,'test_load_ms':load_ms,'experiment_wall_seconds':time.perf_counter()-start,
       'domains':domains,'runtime_runs':runs,
       'scientific_status':{'actual_graph_verifier':'CONDITIONAL','executable_soundness':'NOT ESTABLISHED','formal_certified_reuse':'NOT ESTABLISHED',
           'operational_reuse_count':0,'net_savings':'NEGATIVE for measured fail-closed policies; no claim for all possible regimes',
           'G2':'HOLD','G3':'PARTIAL, numerical obligation unresolved','G4':'NOT ESTABLISHED','G5':'negative measured policy result, not PASS',
           'H21_LipCache':'NOT ESTABLISHED: incompatible strict equivalence target without new trained/verified GuardNet',
           'H24':'NOT ESTABLISHED; evaluated policies do not exhibit combined sound/useful/cost-saving regime'},
       'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'ecdr/week3_verifier.py',out/'protocol.json',out/'geometry.json',out/'domain_trace.csv',out/'runtime_trace.csv']}}
    dump(out/'experiment_results.json',results);print('completed',results['experiment_wall_seconds'],'seconds',flush=True)

if __name__=='__main__':main()
