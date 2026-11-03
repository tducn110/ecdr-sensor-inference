#!/usr/bin/env python3
"""Reproducible, inference-only audit of the frozen Week 1 K=12 SmallCNN1D."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.backbone import SmallCNN1D
from ecdr.data import load_full_manifest_with_splits, load_raw_windows

REF=ROOT/'output/week1_closeout_20261102/reference'
OUT=ROOT/'output/week2_foundations_20261103'
TOL={'logit_atol':1e-5,'logit_rtol':1e-5,'normalization_atol':0.0,'normalization_rtol':0.0}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def dump(p,obj): p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def metrics(y,p,k):
 cm=np.zeros((k,k),dtype=np.int64); np.add.at(cm,(y,p),1)
 tp=np.diag(cm); sup=cm.sum(1); pred=cm.sum(0)
 rec=np.divide(tp,sup,out=np.zeros(k,float),where=sup>0)
 prec=np.divide(tp,pred,out=np.zeros(k,float),where=pred>0)
 f1=np.divide(2*prec*rec,prec+rec,out=np.zeros(k,float),where=(prec+rec)>0)
 return cm,{'n':int(len(y)),'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),'per_class':[{'support':int(sup[i]),'predicted':int(pred[i]),'precision':float(prec[i]),'recall':float(rec[i]),'f1':float(f1[i])} for i in range(k)]}
def main():
 started=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=OUT); args=ap.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1); torch.manual_seed(0)
 cfg=json.loads((REF/'config.json').read_text()); sel=json.loads((REF/'selection.json').read_text()); ckpt_path=REF/'selected_best.pt'; ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
 assert ckpt['run_id']==cfg['run_id']==sel['run_id'] and ckpt['seed']==sel['selected_seed']==43
 assert sha(ckpt_path)==sel['selected_best_sha256']
 model=SmallCNN1D(num_classes=ckpt['num_classes'],cfg=ckpt['model_cfg']); model.load_state_dict(ckpt['state_dict']); model.eval()
 names=ckpt['class_names']; assert names==cfg['class_order']
 params=sum(p.numel() for p in model.parameters() if p.requires_grad); assert params==283692==cfg['expected_params']==ckpt['expected_params']
 # Capture every leaf module output shape, parameter count, and registered buffers.
 records=[]; hooks=[]
 for name,m in model.named_modules():
  if not list(m.children()):
   def hook(mod,ins,outs,name=name):
    if any(r['module']==name for r in records): return
    x=outs[0] if isinstance(outs,(tuple,list)) else outs
    records.append({'module':name,'type':type(mod).__name__,'input_shape':list(ins[0].shape),'output_shape':list(x.shape),'trainable_parameters':sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad),'buffers':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in mod.named_buffers(recurse=False)}})
   hooks.append(m.register_forward_hook(hook))
 # supervised test dataset and full chronological stream are distinct analyses.
 man=load_full_manifest_with_splits(); df=man[man.split=='test'].copy().reset_index(drop=True); raw=load_raw_windows(df.global_row_id.to_numpy(np.int64))
 mean,std=np.load(REF/'normalization.npz').values()
 assert np.array_equal(mean,ckpt['norm_mean']) and np.array_equal(std,ckpt['norm_std'])
 assert np.array_equal(mean,torch.tensor(ckpt['norm_mean']).numpy())
 # Validate normalization preprocessing identity from frozen artifact bytes.
 norm_id=hashlib.sha256(b'norm:train_only:per_channel:axes_0_2:clip_1e-6:float32:'+mean.tobytes()+std.tobytes()).hexdigest()
 assert norm_id==ckpt['preprocessing_id']
 assert sha(REF/'normalization.npz')==ckpt['norm_sha256']
 x=((raw-mean[None,:,None])/std[None,:,None]).astype(np.float32)
 with torch.inference_mode(): model(torch.from_numpy(x[:1]))
 for h in hooks: h.remove()
 stored=np.load(REF/'test_predictions.npz'); assert np.array_equal(stored['global_row_id'],df.global_row_id.to_numpy())
 logits=[]
 with torch.inference_mode():
  for i in range(0,len(x),256): logits.append(model(torch.from_numpy(x[i:i+256])).numpy())
 logits=np.concatenate(logits); pred=logits.argmax(1); sp=stored['predictions']; slog=stored['logits']
 for h in hooks: h.remove()
 max_abs=float(np.max(np.abs(logits-slog))); within=bool(np.allclose(logits,slog,atol=TOL['logit_atol'],rtol=TOL['logit_rtol']))
 assert within and np.array_equal(pred,sp)
 manifest=pd.read_csv(REF/'test_predictions_manifest.csv')
 assert np.array_equal(manifest.global_row_id.to_numpy(),df.global_row_id.to_numpy())
 assert np.array_equal(manifest.predicted_label_zero_based.to_numpy(),pred)
 assert np.array_equal(manifest.predicted_label_name.to_numpy(),np.asarray(names)[pred])
 csv_margin=manifest.winning_point_margin.to_numpy(float); stored_margin=stored['winning_margins'].astype(float)
 csv_margin_max_abs=float(np.max(np.abs(csv_margin-stored_margin)))
 assert np.allclose(csv_margin,stored_margin,atol=1e-5,rtol=1e-5)
 eval_results=json.loads((REF/'evaluation_results.json').read_text())['seed_evaluations']['seed_43']
 # Test margin = top logit minus runner-up, distinct from correct-label margin.
 sorted_logits=np.sort(logits,axis=1); top_margin=sorted_logits[:,-1]-sorted_logits[:,-2]
 stored_margin_error=float(np.max(np.abs(stored_margin-top_margin)))
 manifest_margin_error=float(np.max(np.abs(csv_margin-top_margin)))
 assert np.allclose(stored_margin,top_margin,atol=1e-5,rtol=1e-5)
 assert np.allclose(csv_margin,top_margin,atol=1e-5,rtol=1e-5)
 y=df.label_majority.to_numpy(np.int64)-1; supervised=df.label_policy.isin(['pure','majority']).to_numpy() & (df.label_majority.between(1,12).to_numpy())
 ys=y[supervised]; ps=pred[supervised]; cm,m=metrics(ys,ps,12)
 assert len(records)==15, f'Expected 15 leaf modules, found {len(records)}'
 assert (int(supervised.sum()),float(m['accuracy']),float(m['macro_f1']))==(3572,0.8905375139977604,0.7762645087843447)
 assert abs(float(eval_results['accuracy'])-m['accuracy'])<1e-12
 assert abs(float(eval_results['macro_f1'])-m['macro_f1'])<1e-12
 assert bool(np.array_equal(cm,np.asarray(eval_results['confusion_matrix'],dtype=np.int64)))
 d={'global_test_windows':int(len(df)),'supervised_windows':int(supervised.sum()),'stored_prediction_disagreement_count':int((pred!=sp).sum()),'ground_truth_error_count':int((ps!=ys).sum()),'stored_logits_max_abs_error':max_abs,'stored_logits_allclose':within,'manifest_predictions_exact_match':True,'manifest_winning_margin_vs_stored_max_abs_error':csv_margin_max_abs,'stored_winning_margin_vs_fresh_max_abs_error':stored_margin_error,'manifest_winning_margin_vs_fresh_max_abs_error':manifest_margin_error,'evaluation_results_accuracy_match':bool(abs(float(eval_results['accuracy'])-m['accuracy'])<1e-12),'evaluation_results_macro_f1_match':bool(abs(float(eval_results['macro_f1'])-m['macro_f1'])<1e-12),'tolerance':TOL,'metrics':m,'confusion_matrix':cm.tolist(),'top_two_logit_margin':{'definition':'largest logit minus second-largest logit; nonnegative, prediction confidence proxy, not a correctness guarantee','min':float(top_margin.min()),'p05':float(np.quantile(top_margin,.05)),'p25':float(np.quantile(top_margin,.25)),'median':float(np.median(top_margin)),'p75':float(np.quantile(top_margin,.75)),'p95':float(np.quantile(top_margin,.95)),'max':float(top_margin.max()),'zero_count':int(np.sum(top_margin==0))}}
 # MACs per sample on recorded shapes; Conv MACs excludes bias, BN, ReLU, pooling; Linear excludes bias.
 macs=[]
 for r in records:
  mod=dict(model.named_modules())[r['module']]; sh=r['output_shape']
  if isinstance(mod,nn.Conv1d): v=int(sh[2]*mod.out_channels*(mod.in_channels//mod.groups)*mod.kernel_size[0]); kind='conv multiply-accumulates; excludes bias and BN/activation/pool'
  elif isinstance(mod,nn.Linear): v=int(mod.in_features*mod.out_features); kind='linear multiply-accumulates; excludes bias'
  else: v=0; kind='not counted'
  macs.append({'module':r['module'],'macs_per_sample':v,'convention':kind})
 # RF/jump recurrence for convs and pools in sequence. Conv padding does not change span.
 rf,jump=1,1; layers=[]
 for label,k,s in [('conv1',5,1),('pool1',2,2),('conv2',3,1),('pool2',2,2),('conv3',3,1)]:
  rf += (k-1)*jump; jump*=s; layers.append({'operation':label,'kernel':k,'stride':s,'receptive_field_samples':rf,'input_jump_samples':jump})
 total_macs=sum(z['macs_per_sample'] for z in macs)
 assert total_macs==1172992 and rf==20 and jump==4
 # BatchNorm eval determinism repeated on same already-normalized sample.
 with torch.inference_mode(): a=model(torch.from_numpy(x[:1])).numpy(); b=model(torch.from_numpy(x[:1])).numpy()
 det=bool(np.array_equal(a,b))
 # Forward-only batch-1 timing, resident normalized CPU tensors, separate from input/model hook overhead.
 warm=100; repeats=1000; sample=torch.from_numpy(x[:1].copy())
 with torch.inference_mode():
  for _ in range(warm): model(sample)
  ts=[]
  for _ in range(repeats):
   t=time.perf_counter_ns(); model(sample); ts.append((time.perf_counter_ns()-t)/1e6)
 # architecture comparison count only; different function/classifier, no performance equivalence.
 from ecdr.model import ECDRConv1DNet

 def count_arch(net): return sum(p.numel() for p in net.parameters() if p.requires_grad)
 comparison={'SmallCNN1D_K12':params,'ECDRConv1DNet_K12':count_arch(ECDRConv1DNet(num_classes=12)),'interpretation':'static parameter-count comparison only; distinct architecture/function; not a controlled accuracy or latency comparison'}
 input_files=[ROOT/'scripts/backbone.py',ROOT/'ecdr/data.py',ROOT/'scripts/week1_reference_pipeline.py',ROOT/'data/windows.npz',ROOT/'data/window_manifest.parquet',ROOT/'data/windows_with_split.parquet',REF/'normalization.npz',REF/'test_predictions_manifest.csv',REF/'test_predictions.npz',REF/'evaluation_results.json']
 cpu_model='Unknown'
 try:
  for line in Path('/proc/cpuinfo').read_text().splitlines():
   if line.lower().startswith('model name'):
    cpu_model=line.split(':',1)[1].strip(); break
 except OSError: pass
 verified_checks=['checkpoint_sha256_matches_selection','run_seed_epoch_class_order_match','normalization_arrays_match_checkpoint_exactly','normalization_preprocessing_id_matches','normalization_file_sha256_matches_checkpoint','test_row_ids_match_stored_prediction_artifact','stored_logits_allclose_atol_1e-5_rtol_1e-5','stored_argmax_predictions_exact_match','manifest_row_ids_predictions_and_class_names_exact_match','manifest_and_stored_margins_match_atol_1e-5_rtol_1e-5','stored_margins_match_fresh_top_two_logits_atol_1e-5_rtol_1e-5','manifest_margins_match_fresh_top_two_logits_atol_1e-5_rtol_1e-5','accuracy_macro_f1_and_confusion_match_evaluation_results','parameter_count_is_283692','leaf_module_count_is_15','mac_count_is_1172992','receptive_field_is_20_and_stride_is_4','batchnorm_eval_repeated_forward_is_deterministic']
 modelinfo={'audit_executed_at_utc':started,'audit_script_sha256':sha(Path(__file__)),'verified_check_names':verified_checks,'verified_check_count':len(verified_checks),'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in input_files},'provenance':{'run_id':cfg['run_id'],'selected_seed':ckpt['seed'],'selected_epoch':ckpt['best_epoch'],'checkpoint_sha256':sha(ckpt_path),'expected_checkpoint_sha256':sel['selected_best_sha256'],'normalization_id':norm_id,'test_predictions_sha256':sha(REF/'test_predictions.npz'),'test_predictions_manifest_sha256':sha(REF/'test_predictions_manifest.csv'),'evaluation_results_sha256':sha(REF/'evaluation_results.json'),'config_sha256':sha(REF/'config.json'),'selection_sha256':sha(REF/'selection.json')},'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'cpu_model':cpu_model,'platform':platform.platform(),'machine':platform.machine(),'torch_threads':torch.get_num_threads()},'class_names':names,'input_shape_per_sample':[6,128],'architecture':cfg['model_cfg'],'trainable_parameters':params,'leaf_layers':records,'cost':{'layers':macs,'total_macs_per_sample':total_macs,'convention':'MAC=one multiply plus accumulation; counts convolution/linear only; excludes biases, BN, ReLU, max-pool, memory traffic and softmax'},'receptive_field':{'derivation':'r_next = r + (k-1)*j; j_next=j*s, starting r=j=1; same-padding convolution; valid pool2','layers':layers,'final_receptive_field_samples':rf,'final_stride_samples':jump,'sampling_rate_hz':50,'nominal_temporal_span_seconds':rf/50},'batchnorm_eval_deterministic':det,'timing_ms':{'device':'CPU','torch_threads':1,'batch_size':1,'warmups':warm,'repeats':repeats,'sample_input':'fixed first chronological test window after frozen normalization; model.eval(); resident tensor','p50':float(np.quantile(ts,.5)),'p95':float(np.quantile(ts,.95)),'mean':float(np.mean(ts)),'scope':'forward only on resident pre-normalized tensor; excludes preprocessing, event logic, membership, fallback, data movement and I/O; environment-specific, not speedup evidence'},'candidate_region_status':'not computed; pointwise top-two logit margins are not region bounds or certification','architecture_parameter_comparison':comparison,'metrics':d}
 assert det
 dump(out/'model_audit.json',modelinfo)
 # tables csv
 pd.DataFrame(records).to_csv(out/'model_layer_shapes.csv',index=False)
 pd.DataFrame([{'class_id':i,'class_name':names[i],**m['per_class'][i]} for i in range(12)]).to_csv(out/'model_per_class_metrics.csv',index=False)
 pd.DataFrame(cm,index=names,columns=names).to_csv(out/'model_confusion_matrix.csv')
 pd.DataFrame({'true_class_id':ys,'predicted_class_id':ps,'top_two_logit_margin':top_margin[supervised],'is_correct':ps==ys}).to_csv(out/'model_supervised_test_rows.csv',index=False)
 fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(cm,cmap='Blues'); ax.set_xticks(range(12),names,rotation=65,ha='right'); ax.set_yticks(range(12),names); ax.set_xlabel('Predicted'); ax.set_ylabel('Ground truth'); ax.set_title('K=12 supervised test confusion matrix'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/'model_confusion_matrix.png',dpi=170); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5)); ax.hist(top_margin,bins=40,color='#3478a8',edgecolor='white'); ax.set_xlabel('Top logit minus runner-up'); ax.set_ylabel('Test windows (full 5,053 stream)'); ax.set_title('Frozen reference decision margin'); fig.tight_layout(); fig.savefig(out/'model_top_two_margins.png',dpi=170); plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar([z['module'] for z in macs],[z['macs_per_sample'] for z in macs],color='#5d8a66'); ax.set_ylabel('MACs per sample'); ax.set_title('Conv/Linear arithmetic by layer'); ax.tick_params(axis='x',rotation=30); fig.tight_layout(); fig.savefig(out/'model_layer_macs.png',dpi=170); plt.close(fig)
 print(json.dumps({'output':str(out),'checkpoint_sha256':sha(ckpt_path),'parameters':params,'macs':modelinfo['cost']['total_macs_per_sample'],'rf_samples':rf,'stride':jump,'full_test':len(df),'supervised_test':int(supervised.sum()),'disagreement':d['stored_prediction_disagreement_count'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'margin':d['top_two_logit_margin'],'timing_ms':modelinfo['timing_ms'],'artifacts':sorted(p.name for p in out.iterdir())},indent=2))
if __name__=='__main__': main()
