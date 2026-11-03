#!/usr/bin/env python3
"""Actual frozen K=12 auto-LiRPA capability probe; never an executable certificate.

The optional singleton-height transform is algebraically equivalent in reals:
I(z)[n,c,0,t]=z[n,c,t], with kernel (1,k), padding (0,p), and pool (1,2).
It preserves summation indices, BN channel constants, maxima and flatten order.
Neither that argument nor sampled agreement establishes float32 equivalence.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backbone import SmallCNN1D


class SingletonHeightCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        layers=[]
        for module in model.conv:
            if isinstance(module, nn.Conv1d):
                m=nn.Conv2d(module.in_channels,module.out_channels,(1,module.kernel_size[0]),
                            padding=(0,module.padding[0]),bias=module.bias is not None)
                m.weight.data.copy_(module.weight.data.unsqueeze(2))
                if m.bias is not None: m.bias.data.copy_(module.bias.data)
            elif isinstance(module, nn.BatchNorm1d):
                m=nn.BatchNorm2d(module.num_features,eps=module.eps)
                m.load_state_dict(module.state_dict())
            elif isinstance(module, nn.MaxPool1d): m=nn.MaxPool2d((1,2),stride=(1,2))
            elif isinstance(module, nn.ReLU): m=nn.ReLU()
            else: raise TypeError(type(module).__name__)
            layers.append(m)
        import copy
        self.conv=nn.Sequential(*layers);self.head=copy.deepcopy(model.head)
        self.eval()
    def forward(self,x): return self.head(self.conv(x))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=ROOT/'output/week3_certification_20261103');args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    ref=ROOT/'output/week1_closeout_20261102/reference'
    ckpt=torch.load(ref/'selected_best.pt',map_location='cpu',weights_only=False)
    model=SmallCNN1D(num_classes=12,cfg=ckpt['model_cfg']);model.load_state_dict(ckpt['state_dict']);model.eval()
    from ecdr.data import load_full_stream
    raw,man=load_full_stream('val');norm=np.load(ref/'normalization.npz')
    z=((raw-norm['mean'][None,:,None])/norm['std'][None,:,None]).astype(np.float32)
    # Fixed evenly-spaced validation anchors; no test tuning.
    ids=np.linspace(0,len(z)-1,8,dtype=int)
    surrogate=SingletonHeightCNN(model)
    with torch.no_grad():
        a=model(torch.from_numpy(z[ids]));b=surrogate(torch.from_numpy(z[ids]).unsqueeze(2))
    report={'torch':torch.__version__,'auto_lirpa':importlib.metadata.version('auto-lirpa'),
            'requirements':importlib.metadata.requires('auto-lirpa'),
            'version_contract_satisfied':False,'validation_indices':ids.tolist(),
            'checkpoint_sha256':hashlib.sha256((ref/'selected_best.pt').read_bytes()).hexdigest(),
            'singleton_height_equivalence':{'real_algebra':'DERIVED operator-by-operator; see module docstring',
              'sampled_logit_max_abs_error':float((a-b).abs().max()),
              'sampled_decision_disagreements':int((a.argmax(1)!=b.argmax(1)).sum()),
              'executable_universal_equivalence':'NOT ESTABLISHED'},'probes':[]}
    try:
        from auto_LiRPA import BoundedModule,BoundedTensor,PerturbationLpNorm
    except Exception:
        report['import_error']=traceback.format_exc();(out/'lirpa_probe.json').write_text(json.dumps(report,indent=2)+'\n');return
    for name,net,base in [('native_1d',model,torch.from_numpy(z[ids[:1]])),
                          ('singleton_height_2d',surrogate,torch.from_numpy(z[ids[:1]]).unsqueeze(2))]:
        start=time.perf_counter();r={'graph':name,'bounds':[]}
        try:
            bounded=BoundedModule(net,base,bound_opts={'conv_mode':'matrix'},device='cpu');bounded.eval()
            r['conversion']='PASS';r['conversion_ms']=(time.perf_counter()-start)*1000
            for i in range(2 if name=='singleton_height_2d' else 1):
                x=torch.from_numpy(z[ids[i:i+1]])
                if name=='singleton_height_2d':x=x.unsqueeze(2)
                with torch.no_grad():c=int(net(x).argmax(1))
                others=[j for j in range(12) if j!=c]
                C=torch.zeros(1,11,12);C[0,:,c]=1;C[0,torch.arange(11),others]=-1
                for radius in [0.0,0.001,0.01]:
                    bt=BoundedTensor(x,PerturbationLpNorm(norm=np.inf,x_L=x-radius,x_U=x+radius))
                    for method in ['IBP','CROWN']:
                        t=time.perf_counter()
                        try:
                            lo,hi=bounded.compute_bounds(x=(bt,),C=C,method=method)
                            r['bounds'].append({'anchor_val_index':int(ids[i]),'radius':radius,'method':method,
                               'lower_margins':lo.detach().numpy().reshape(-1).tolist(),'competitors':others,
                               'elapsed_ms':(time.perf_counter()-t)*1000,'status':'CONDITIONAL_DIAGNOSTIC',
                               'all_positive':bool(torch.isfinite(lo).all() and (lo>0).all())})
                        except Exception as e:
                            r['bounds'].append({'radius':radius,'method':method,'status':'FAILED','exception':repr(e),
                               'elapsed_ms':(time.perf_counter()-t)*1000})
        except Exception as e:r['conversion']='FAILED';r['exception']=repr(e);r['traceback']=traceback.format_exc()
        r['total_ms']=(time.perf_counter()-start)*1000;report['probes'].append(r)
        print(name,r.get('conversion'),len(r['bounds']),flush=True)
    report['formal_runtime_reuse']='DISABLED: version mismatch, bound arithmetic and reference/surrogate floating-point equivalence unproved'
    (out/'lirpa_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
