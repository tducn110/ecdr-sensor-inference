#!/usr/bin/env python3
"""Independent descriptive audit of frozen HAPT windows and SmallCNN1D inputs.

Writes only under output/week2_foundations_20261103. Does not train or mutate
the frozen Week 1 reference artifacts. Rebuilds windows from raw HAPT files.
"""
from __future__ import annotations
import hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/week2_foundations_20261103'
RAW=ROOT/'data/hapt/RawData'
W,S=128,64

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 started_at_utc=datetime.now(timezone.utc).isoformat()
 OUT.mkdir(parents=True,exist_ok=True)
 man=pd.read_parquet(ROOT/'data/window_manifest.parquet')
 spl=pd.read_parquet(ROOT/'data/windows_with_split.parquet')
 with np.load(ROOT/'data/windows.npz') as z: x=z['x']
 labels=np.loadtxt(RAW/'labels.txt',dtype=np.int64)
 byrec={}
 for r in labels: byrec.setdefault(int(r[0]),[]).append(r)
 rebuilt=[]; rows=[]; offset_diffs=0
 for ap in sorted(RAW.glob('acc_exp*_user*.txt')):
  stem=ap.name[4:]; gp=RAW/('gyro_'+stem)
  import re
  exp=int(re.search(r'exp(\d+)',stem).group(1)); usr=int(re.search(r'user(\d+)',stem).group(1))
  a=np.loadtxt(ap,dtype=np.float32).reshape(-1,3); g=np.loadtxt(gp,dtype=np.float32).reshape(-1,3)
  sig=np.concatenate([a,g],axis=1).T; T=sig.shape[1]
  pl=np.zeros(T+1,dtype=np.int16)
  pl_onebased=np.zeros(T+1,dtype=np.int16)
  for r in byrec.get(exp,[]): pl[int(r[3]):int(r[4])+1]=int(r[2])
  for r in byrec.get(exp,[]): pl_onebased[max(0,int(r[3])-1):int(r[4])]=int(r[2])
  local=0
  for st in range(0,T-W+1,S):
   en=st+W; labs=pl[st:en]; ann=labs[labs>0]
   if len(ann)==0: pol,maj='unlabeled',0
   else:
    vals,cnts=np.unique(ann,return_counts=True); maj=int(vals[np.argmax(cnts)]); frac=int(cnts.max())/W
    pol,maj=('pure',maj) if len(vals)==1 and frac==1 else (('majority',maj) if frac>=.6 else ('ambiguous',maj))
   alabs=pl_onebased[st:en]; aann=alabs[alabs>0]
   if len(aann)==0: apol,amaj='unlabeled',0
   else:
    av,ac=np.unique(aann,return_counts=True); amaj=int(av[np.argmax(ac)]); af=int(ac.max())/W
    apol,amaj=('pure',amaj) if len(av)==1 and af==1 else (('majority',amaj) if af>=.6 else ('ambiguous',amaj))
   offset_diffs += int((pol,maj)!=(apol,amaj))
   rec={'exp':exp,'subject':usr,'win_idx':local,'start':st,'end':en,'label_policy':pol,'label_majority':maj,
        'n_distinct_labels':int(len(np.unique(ann))), 'annotated_frac':round(len(ann)/W,4),
        'crosses_boundary':bool(len(np.unique(ann))>1)}
   rows.append(rec); rebuilt.append(sig[:,st:en]); local+=1
 rebuilt=np.stack(rebuilt).astype(np.float32); rm=pd.DataFrame(rows)
 # compare aligned rows by stable position, exact numeric values
 raw_equal=bool(np.array_equal(rebuilt,x)); manifest_equal=bool(rm.equals(man.reset_index(drop=True)))
 # Annotated ranges use inclusive end in labels.txt; compare documented range length and assignment count.
 # count total sample assignments and overlaps from raw label rows
 assignments=int(sum(int(r[4])-int(r[3])+1 for r in labels))
 # pair consecutive window rows within recording; raw sensor and normalized aligned distance
 df=man.copy(); df['global_row_id']=np.arange(len(df)); df=df.merge(spl,on=['exp','win_idx'],validate='one_to_one',suffixes=('','_split'))
 pairs=[]; distances=[]; samepos_by_pair=[]
 mean=np.load(ROOT/'output/week1_closeout_20261102/reference/normalization.npz')['mean'].astype(np.float32)
 std=np.load(ROOT/'output/week1_closeout_20261102/reference/normalization.npz')['std'].astype(np.float32)
 xn=((x-mean[None,:,None])/std[None,:,None]).astype(np.float32)
 for _,grp in df.groupby('exp',sort=False):
  ids=grp.sort_values('win_idx').global_row_id.to_numpy()
  for i,j in zip(ids[:-1],ids[1:]):
   if df.loc[i,'start']+S!=df.loc[j,'start']: continue
   pairs.append((int(i),int(j)))
   # Distances compare both windows at identical original sample coordinates.
   distances.append([float(np.mean(np.abs(x[i,:,S:]-x[j,:,:S]))),float(np.mean(np.abs(xn[i,:,S:]-xn[j,:,:S])))])
   samepos_by_pair.append([np.mean(np.abs(x[i]-x[j]),axis=1),np.mean(np.abs(xn[i]-xn[j]),axis=1)])
 pairarr=np.asarray(pairs,dtype=np.int64); darr=np.asarray(distances)
 samepos_by_pair=np.asarray(samepos_by_pair,dtype=np.float64) # pair x raw/norm x channel
 # Support counts, subject disjointness, per-split stream and supervised support
 splits={s:df[df.split==s] for s in ['train','val','test']}
 subject_sets={s:set(g.subject.unique().tolist()) for s,g in splits.items()}
 support={}
 for s,g in splits.items():
  valid=g.label_policy.isin(['pure','majority']) & g.label_majority.between(1,12)
  support[s]={'stream_n':len(g),'supervised_n':int(valid.sum()),'excluded_ambiguous':int((g.label_policy=='ambiguous').sum()),
              'excluded_unlabeled':int((g.label_policy=='unlabeled').sum()),
              'by_class':[{'id':c,'n':int(((g.label_policy.isin(['pure','majority']))&(g.label_majority==c)).sum()),
                            'subjects':int(g.loc[(g.label_policy.isin(['pure','majority']))&(g.label_majority==c),'subject'].nunique())} for c in range(1,13)]}
 # Shared-overlap comparison is a reconstruction check (identical source samples by design).
 # Also compare each adjacent window at the same local coordinates, and raw recording lag correlations.
 d=darr[:,0]
 lagcorr={64:[],128:[]}
 import re
 for ap in sorted(RAW.glob('acc_exp*_user*.txt')):
  stem=ap.name[4:]; gp=RAW/('gyro_'+stem)
  a=np.loadtxt(ap,dtype=np.float32).reshape(-1,3); g=np.loadtxt(gp,dtype=np.float32).reshape(-1,3)
  sig=np.concatenate([a,g],axis=1)
  for lag in (64,128):
   for ch in range(6): lagcorr[lag].append(float(np.corrcoef(sig[:-lag,ch],sig[lag:,ch])[0,1]))
 chan_names=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
 samepos_summary={}
 for channel_idx,name in enumerate(chan_names):
  raw=samepos_by_pair[:,0,channel_idx]; norm=samepos_by_pair[:,1,channel_idx]
  samepos_summary[name]={'raw_mean_abs_distance_mean':float(raw.mean()),'raw_mean_abs_distance_median':float(np.median(raw)),
   'raw_mean_abs_distance_p95':float(np.quantile(raw,.95)),'normalized_mean_abs_distance_mean':float(norm.mean()),
   'normalized_mean_abs_distance_median':float(np.median(norm)),'normalized_mean_abs_distance_p95':float(np.quantile(norm,.95))}
 overlap={'aligned_pair_count':len(pairarr),'raw_abs_distance_mean':float(d.mean()),'raw_abs_distance_median':float(np.median(d)),
          'raw_abs_distance_p95':float(np.quantile(d,.95)),'raw_exact_equal_pairs':int(np.sum(d==0)),
          'normalized_abs_distance_mean':float(darr[:,1].mean()),
          'same_local_coordinate_mae_by_channel':samepos_summary,
          'recording_channel_lag64_correlation_mean':float(np.mean(lagcorr[64])),
          'recording_channel_lag64_correlation_by_channel':{chan_names[i]:float(np.mean(lagcorr[64][i::6])) for i in range(6)},
          'recording_channel_lag128_correlation_mean':float(np.mean(lagcorr[128])),
          'recording_channel_lag128_correlation_by_channel':{chan_names[i]:float(np.mean(lagcorr[128][i::6])) for i in range(6)},
          'interpretation':'the aligned 64-sample overlap is exactly shared input; same-local-coordinate adjacent windows differ. Lag correlations are descriptive across within-recording sensor samples. Overlapping windows are dependent; these summaries are not iid evidence or confidence intervals.'}
 # Save pair list for reproducibility with indices and recording identity
 pair_csv={'left_global_row_id':pairarr[:,0],'right_global_row_id':pairarr[:,1],
  'exp':df.iloc[pairarr[:,0]].exp.to_numpy(),'left_win_idx':df.iloc[pairarr[:,0]].win_idx.to_numpy(),
  'right_win_idx':df.iloc[pairarr[:,1]].win_idx.to_numpy(),'raw_mean_abs_distance':darr[:,0],
  'normalized_mean_abs_distance':darr[:,1]}
 for i,name in enumerate(chan_names):
  pair_csv[f'same_local_raw_mae_{name}']=samepos_by_pair[:,0,i]
  pair_csv[f'same_local_normalized_mae_{name}']=samepos_by_pair[:,1,i]
 pd.DataFrame(pair_csv).to_csv(OUT/'data_aligned_overlap_pairs.csv',index=False)
 nfile=ROOT/'output/week1_closeout_20261102/reference/normalization.npz'
 normj=json.loads((nfile.parent/'normalization.json').read_text())
 ckpt=torch.load(nfile.parent/'selected_best.pt',map_location='cpu',weights_only=False)
 # independently derive train-only norm using supervised training indices
 tr=splits['train']; mask=tr.label_policy.isin(['pure','majority'])&tr.label_majority.between(1,12)
 tx=x[tr.loc[mask,'global_row_id'].to_numpy()]
 m=tx.mean(axis=(0,2),dtype=np.float64).astype(np.float32); sd=tx.std(axis=(0,2),dtype=np.float64).astype(np.float32); sd=np.clip(sd,1e-6,None)
 checks={
  'raw_rebuilt_windows_exact':raw_equal,'raw_rebuilt_manifest_exact':manifest_equal,
  'window_shape':list(x.shape),'window_dtype':str(x.dtype),'finite_values':bool(np.isfinite(x).all()),
  'window_stride':64,'window_length':128,'recordings':int(man.exp.nunique()),'subjects':int(man.subject.nunique()),
  'label_policy_counts':{k:int(v) for k,v in man.label_policy.value_counts().items()},
  'split_stream_denominators':{s:int(len(g)) for s,g in splits.items()},'split_supervised_support':support,
  'subject_counts_by_split':{s:len(v) for s,v in subject_sets.items()},
  'subject_disjoint':not any(subject_sets[a]&subject_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]),
  'recordings_single_split':bool(df.groupby('exp').split.nunique().eq(1).all()),
  'row_count_17453':len(man)==17453,'expected_full_stream_split_counts':{ 'train':9978,'val':2422,'test':5053},
  'raw_label_rows':int(len(labels)),'raw_label_assignment_sample_total_inclusive_ranges':assignments,
  'one_sample_coordinate_shift_changed_window_policy_or_label_assignments':int(offset_diffs),
  'raw_label_boundary_convention_caveat':'Builder assigns labels with plabel[start:end+1] using integer coordinates directly. The one-sample alternate convention changes 98 window policy/label assignments; this audit does not claim which upstream coordinate convention is semantically intended.',
  'normalization':{'train_supervised_n':int(mask.sum()),'recomputed_mean':m.tolist(),'recomputed_std':sd.tolist(),
   'stored_mean':normj['mean'],'stored_std':normj['std'],'mean_exact':bool(np.array_equal(m,np.asarray(normj['mean'],np.float32))),
   'std_exact':bool(np.array_equal(sd,np.asarray(normj['std'],np.float32))),'stored_preprocessing_id':normj['preprocessing_id'],
   'checkpoint_seed':int(ckpt['seed']),'checkpoint_best_epoch':int(ckpt['best_epoch']),
   'checkpoint_best_val_macro_f1':float(ckpt['best_val_macro_f1']),
   'checkpoint_norm_matches_saved_json':bool(np.array_equal(np.asarray(ckpt['norm_mean']),np.asarray(normj['mean'],np.float32)) and np.array_equal(np.asarray(ckpt['norm_std']),np.asarray(normj['std'],np.float32))),
   'checkpoint_preprocessing_id_matches':ckpt['preprocessing_id']==normj['preprocessing_id'],
   'scope':'train-only supervised rows, axes (N,time), float64 accumulation -> float32, std floor 1e-6'},
  'overlap':overlap,
  'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/03_build_windows.py',ROOT/'scripts/ecdr_audit_support.py',ROOT/'ecdr/data.py',ROOT/'ecdr/preprocessing.py',ROOT/'data/hapt/RawData/labels.txt',ROOT/'data/windows.npz',ROOT/'data/window_manifest.parquet',ROOT/'data/windows_with_split.parquet',nfile,nfile.parent/'selected_best.pt']+sorted(RAW.glob('acc_exp*_user*.txt'))+sorted(RAW.glob('gyro_exp*_user*.txt'))},
  'execution':{'started_at_utc':started_at_utc,'completed_at_utc':None,
   'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'platform':platform.platform(),
   'script_sha256':sha(Path(__file__))},
  'evidence_boundary':'This audit was executed now; Week 1 receipts/checkpoints are stored artifacts. No model inference, retraining, certificate, or statistical CI was executed/generated.'}
 expected_policy={'pure':10361,'majority':1917,'ambiguous':1588,'unlabeled':3587}
 expected_splits={'train':9978,'val':2422,'test':5053}
 assert x.shape==(17453,6,128) and x.dtype==np.float32 and np.isfinite(x).all(), 'raw tensor shape/dtype/finiteness failed'
 assert raw_equal, 'raw window rebuild differs from frozen tensor'
 assert manifest_equal, 'rebuilt manifest differs from frozen manifest'
 assert checks['label_policy_counts']==expected_policy, f"label policy counts differ: {checks['label_policy_counts']}"
 assert checks['split_stream_denominators']==expected_splits, f"split denominators differ: {checks['split_stream_denominators']}"
 assert checks['subject_counts_by_split']=={'train':17,'val':4,'test':9} and checks['subject_disjoint'], 'subject split integrity failed'
 assert checks['recordings_single_split'], 'recording crosses split boundary'
 assert sum(v['supervised_n'] for v in support.values())==12278, 'supervised total differs from expected 12,278'
 assert checks['one_sample_coordinate_shift_changed_window_policy_or_label_assignments']==98, 'one-sample offset delta differs from 98'
 assert len(pairarr)==17392, f'expected 17,392 adjacent aligned pairs, got {len(pairarr)}'
 assert np.array_equal(x[pairarr[:,0],:,S:],x[pairarr[:,1],:,:S]), 'raw shared-overlap samples differ'
 assert np.array_equal(xn[pairarr[:,0],:,S:],xn[pairarr[:,1],:,:S]), 'normalized shared-overlap samples differ'
 assert checks['normalization']['mean_exact'] and checks['normalization']['std_exact'], 'recomputed normalization differs from frozen values'
 assert checks['normalization']['checkpoint_norm_matches_saved_json'] and checks['normalization']['checkpoint_preprocessing_id_matches'], 'checkpoint normalization contract mismatch'
 # historical support file summarized without treating it as current evidence
 hist=json.loads((ROOT/'data/ecdr_12_class_support.json').read_text())
 from ecdr.data import audit_12_class_support
 source_support=audit_12_class_support()
 histnote={'source':'data/ecdr_12_class_support.json','meaning':'historical support snapshot; current source scripts/ecdr_audit_support.py calls ecdr.data.audit_12_class_support, whose majority predicate includes only label_policy in pure/majority and matching class ID. The stored historical counts do not reproduce that current predicate on this frozen manifest, so historical artifact provenance/predicate cannot be confirmed from the artifact alone.',
  'current_source_predicate':'split == selected split AND label_policy in {pure, majority} AND label_majority == class_id',
  'current_source_majority_train_val_test':{k:sum(c[k+'_count'] for c in source_support['majority']['classes']) for k in ['train','val','test']},
  'pure_train_val_test':{k:sum(c[k+'_count'] for c in hist['pure']['classes']) for k in ['train','val','test']},
  'majority_train_val_test':{k:sum(c[k+'_count'] for c in hist['majority']['classes']) for k in ['train','val','test']},
  'current_supervised_train_val_test':{s:v['supervised_n'] for s,v in support.items()},
  'observed_ambiguous_train_val_test':{s:v['excluded_ambiguous'] for s,v in support.items()},
  'historical_majority_matches_current_source':{k:sum(c[k+'_count'] for c in hist['majority']['classes'])==sum(c[k+'_count'] for c in source_support['majority']['classes']) for k in ['train','val','test']}}
 checks['execution']['completed_at_utc']=datetime.now(timezone.utc).isoformat()
 (OUT/'data_audit.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
 (OUT/'data_support_comparison.json').write_text(json.dumps(histnote,indent=2)+'\n')
 print(json.dumps({'output':str(OUT),'checks':{k:checks[k] for k in ['raw_rebuilt_windows_exact','raw_rebuilt_manifest_exact','window_shape','label_policy_counts','split_stream_denominators','subject_counts_by_split','subject_disjoint','raw_label_rows','raw_label_assignment_sample_total_inclusive_ranges']},'overlap':overlap},indent=2))

if __name__=='__main__': main()
