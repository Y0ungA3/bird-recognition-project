import json, platform
from pathlib import Path
import torch
from torchvision import models
root=Path(__file__).resolve().parent
assert torch.cuda.is_available(), 'CUDA is unavailable'
device=torch.device('cuda')
model=models.resnet18(weights=None).to(device)
x=torch.randn(4,3,224,224,device=device)
with torch.autocast('cuda'):
    loss=model(x).square().mean()
loss.backward(); torch.cuda.synchronize()
result={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'compiled_architectures':torch.cuda.get_arch_list(),'resnet18_forward_backward':True,'loss':loss.item()}
(root/'gpu_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
