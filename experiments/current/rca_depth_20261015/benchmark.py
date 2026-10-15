import sys, json, time, statistics, hashlib
from pathlib import Path
import torch, yaml
D=Path(__file__).resolve().parent; ROOT=D.parents[2]
sys.path.insert(0,str(ROOT/'SkateFormer'))
from model.SkateFormer import SkateFormer
torch.set_num_threads(1); torch.manual_seed(1)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True
assert torch.cuda.is_available()
models={}; result={'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__},
'protocol':{'shape':[1,3,64,14,1],'dtype':'FP32','tf32':False,'cpu_threads':1,'warmup':80,'rounds':12,'iterations':60,'random_weights':True,'timing':'synchronized wall clock, GPU resident input; excludes loading/H2D'},
'models':{},'order':[]}
for label,depths in [('B0',[2,2,2,2]),('Current',[2,2,2,2]),('S1',[1,2,2,2]),('S2',[2,1,2,2]),('S3',[2,2,1,2]),('S4',[2,2,2,1])]:
    name='baseline_14p_1person' if label=='B0' else 'rca_14p_1person_ffn4_attndrop01_localt3'
    cfg=yaml.safe_load((ROOT/'SkateFormer/config/train/acceleration'/(name+'.yaml')).read_text())
    cfg['model']='model.SkateFormer.SkateFormer'
    cfg['model_args'].update(depths=depths,channels=[96,192,192,192],embed_dim=96)
    if label.startswith('S'): cfg['work_dir']+='_depth'+''.join(map(str,depths))
    (D/(label+'.candidate.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False))
    torch.manual_seed(1)
    m=SkateFormer(**cfg['model_args'])
    assert [len(s.blocks) for s in m.stages]==depths
    assert all(m.stages[i].blocks[0].downsample is not None for i in [1,2,3])
    m.train(); x=torch.randn(2,3,64,14,1); t=torch.linspace(-1,1,64).repeat(2,1)
    out=m(x,t); assert out.shape==(2,60) and torch.isfinite(out).all()
    torch.nn.functional.cross_entropy(out,torch.tensor([0,1])).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    torch.manual_seed(1); m=SkateFormer(**cfg['model_args']).eval().cuda()
    models[label]=m
    result['models'][label]={'depths':depths,'parameters':sum(p.numel() for p in m.parameters()),'validation':'finite CPU train forward/backward; stage downsamplers retained','rounds':[]}
x=torch.randn(1,3,64,14,1,device='cuda'); t=torch.linspace(-1,1,64,device='cuda')[None]
with torch.no_grad():
    for m in models.values():
        for _ in range(80): y=m(x,t)
        assert y.shape==(1,60) and torch.isfinite(y).all()
    labels=list(models)
    for rnd in range(12):
        offset=rnd%len(labels); order=labels[offset:]+labels[:offset]
        if rnd>=len(labels): order=order[::-1]
        result['order'].append(order)
        for label in order:
            m=models[label]
            for _ in range(10): m(x,t)
            times=[]
            for _ in range(60):
                torch.cuda.synchronize(); start=time.perf_counter()
                m(x,t); torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            result['models'][label]['rounds'].append(times)
        print('Round',rnd+1,{k:round(statistics.mean(v['rounds'][-1]),3) for k,v in result['models'].items()},flush=True)
for label,v in result['models'].items():
    ts=sorted(x for rr in v['rounds'] for x in rr)
    v['summary']={'mean_ms':statistics.mean(ts),'median_ms':statistics.median(ts),'p95_ms':ts[int(.95*(len(ts)-1))],'round_means':[statistics.mean(rr) for rr in v['rounds']]}
result['source_sha256']={}
for rel in ['SkateFormer/model/SkateFormer.py','SkateFormer/model/operators/skate_rca_block.py']:
    raw=(ROOT/rel).read_bytes(); (D/(Path(rel).stem+'.source.py')).write_bytes(raw)
    result['source_sha256'][rel]=hashlib.sha256(raw).hexdigest()
dest=D/'results.json'; assert not dest.exists(); dest.write_text(json.dumps(result,indent=2))
print(json.dumps({k:dict(parameters=v['parameters'],**v['summary']) for k,v in result['models'].items()},indent=2),flush=True)
