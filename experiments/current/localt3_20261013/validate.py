import sys
from pathlib import Path
import torch, yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'SkateFormer'))
from model.SkateFormer import SkateFormer_
torch.set_num_threads(1)
torch.manual_seed(1)
d=ROOT/'SkateFormer/config/train/acceleration'
old=yaml.safe_load((d/'rca_14p_1person_ffn4_attndrop01.yaml').read_text())
new=yaml.safe_load((d/'rca_14p_1person_ffn4_attndrop01_localt3.yaml').read_text())
assert new['model_args'].pop('rca_local_temporal_kernel')==3
old.pop('work_dir'); new.pop('work_dir')
assert old==new, 'Unexpected training protocol difference'
args=old['model_args']
baseline=SkateFormer_(**args).eval()
model=SkateFormer_(**args,rca_local_temporal_kernel=3).eval()
missing,unexpected=model.load_state_dict(baseline.state_dict(),strict=False)
assert not unexpected and all('local_temporal.weight' in k or 'local_scale' in k for k in missing)
assert not any(('local_temporal.weight' in k or 'local_scale' in k) for k in baseline.state_dict())
blocks=[m for m in model.modules() if getattr(m,'local_temporal',None) is not None]
assert len(blocks)==8
x=torch.randn(2,3,64,14,1); t=torch.linspace(-1,1,64).repeat(2,1)
with torch.no_grad():
    for b in blocks: b.local_scale.zero_()
    torch.testing.assert_close(model(x,t),baseline(x,t),rtol=0,atol=0)
    for b in blocks: b.local_scale.fill_(0.1)
model.train()
out=model(x,t)
assert out.shape==(2,60) and torch.isfinite(out).all()
torch.nn.functional.cross_entropy(out,torch.tensor([1,2])).backward()
for b in blocks:
    for p in [b.local_scale,b.local_temporal.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
print('PASS: only intended config differences; 8 branches; scale=0 matches old RCA exactly; forward/backward finite; new weights and scales get nonzero gradients.')
print('Parameters:',sum(p.numel() for p in model.parameters()))

