import importlib.util, importlib, sys, json, time, statistics, hashlib
from pathlib import Path
import torch, yaml
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
sys.path.insert(0,str(ROOT/'SkateFormer'))
network=importlib.import_module('model.SkateFormer')
def module(name):
    spec=importlib.util.spec_from_file_location(name,D/(name+'.py'))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
old=module('before'); opt=module('optimized')
torch.set_num_threads(1)
torch.manual_seed(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True
assert torch.cuda.is_available()
validation=[]
for device in ['cpu','cuda']:
    for length,anchors,local,rel in [(64,16,3,True),(32,16,0,True),(8,16,3,True),(17,7,3,False)]:
        kw=dict(in_channels=32,num_points=14,relation_rank=2,temporal_anchors=anchors,
                local_temporal_kernel=local,rel=rel,attn_drop=0,ffn_ratio=4)
        a=old.SkateRCABlock(**kw).to(device); b=opt.SkateRCABlock(**kw).to(device)
        b.load_state_dict(a.state_dict(),strict=True)
        x=torch.randn(2,32,length,14,device=device,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        oa=a(x); ob=b(y)
        torch.testing.assert_close(oa,ob,rtol=2e-5,atol=2e-6)
        g=torch.randn_like(oa); (oa*g).sum().backward(); (ob*g).sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=2e-4,atol=2e-5)
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb
            if pa.grad is not None:
                torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-3,atol=2e-4)
        validation.append(dict(device=device,T=length,anchors=anchors,local=local,rel=rel,
                               max_abs_output=float((oa-ob).abs().max())))
# Cache must be rebuilt for a different length and excluded from checkpoint.
b=opt.SkateRCABlock(32,num_points=14)
for t in [64,8,32]:
    with torch.no_grad(): b(torch.randn(1,32,t,14))
    assert b._local_distance.shape==(t,16)
assert not any('_local_distance' in k for k in b.state_dict())
models={}; result={'validation':validation,'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__},
'protocol':{'shape':[1,3,64,14,1],'dtype':'float32','TF32':False,'random_weights':True,
'cpu_threads':1,'warmup':80,'rounds':8,'iterations':100,'timing':'synchronized wall clock including Python, GPU-resident input; no H2D/data loader'},
'models':{},'order':[]}
for label,cls in [('B0',None),('Before',old.SkateRCABlock),('Optimized',opt.SkateRCABlock),('FFN3',old.SkateRCABlock)]:
    cfgname='baseline_14p_1person' if label=='B0' else 'rca_14p_1person_ffn4_attndrop01_localt3'
    raw=(ROOT/'SkateFormer/config/train/acceleration'/(cfgname+'.yaml')).read_bytes()
    if label=='FFN3': raw=(D/'ffn3.candidate.yaml').read_bytes()
    (D/(label+'.config.yaml')).write_bytes(raw)
    if cls: network.SkateRCABlock=cls
    torch.manual_seed(1)
    model=network.SkateFormer_(**yaml.safe_load(raw)['model_args']).eval()
    if label=='Before': state={k:v.clone() for k,v in model.state_dict().items()}
    if label=='Optimized': model.load_state_dict(state,strict=True)
    xx=torch.randn(1,3,64,14,1); tt=torch.linspace(-1,1,64)[None]
    with torch.no_grad():
        model(xx,tt)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as prof:
            model(xx,tt)
    counts={e.key:{'count':e.count,'self_cpu_memory_usage':e.self_cpu_memory_usage}
            for e in prof.key_averages() if e.key in ['aten::clone','aten::copy_','aten::contiguous','aten::linspace','aten::bmm']}
    result['models'][label]={'params':sum(p.numel() for p in model.parameters()),'cpu_profile':counts,'rounds':[]}
    models[label]=model.cuda()
x=torch.randn(1,3,64,14,1,device='cuda'); t=torch.linspace(-1,1,64,device='cuda')[None]
with torch.no_grad():
    ref=models['Before'](x,t)
    for label in ['Optimized']:
        val=models[label](x,t)
        torch.testing.assert_close(val,ref,rtol=2e-5,atol=2e-6)
        result['models'][label]['full_model_max_abs_diff']=float((val-ref).abs().max())
    for model in models.values():
        for _ in range(80): model(x,t)
    labels=list(models)
    for round in range(8):
        order=labels[round%4:]+labels[:round%4]
        if round>=4: order=order[::-1]
        result['order'].append(order)
        for label in order:
            model=models[label]
            for _ in range(10): model(x,t)
            times=[]
            for _ in range(100):
                torch.cuda.synchronize(); start=time.perf_counter()
                model(x,t)
                torch.cuda.synchronize(); times.append(1000*(time.perf_counter()-start))
            result['models'][label]['rounds'].append(times)
            print(round+1,label,round_ms:=statistics.mean(times),flush=True)
for label,entry in result['models'].items():
    times=sorted(t for rr in entry['rounds'] for t in rr)
    entry['summary']={'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),
                      'p95_ms':times[int(.95*(len(times)-1))],
                      'round_means':[statistics.mean(rr) for rr in entry['rounds']]}
result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}
dest=D/'results.json'; assert not dest.exists()
dest.write_text(json.dumps(result,indent=2))
print('VALIDATION PASSED',validation,flush=True)
print(json.dumps({k:v['summary'] for k,v in result['models'].items()},indent=2),flush=True)
