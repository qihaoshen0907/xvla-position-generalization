"""Rank-8 LoRA only; original model and domain embeddings stay frozen."""
from common import *
import math
from torch import nn
from torch.nn import functional as F
class LoRALinear(nn.Module):
 def __init__(self,base,r=8,alpha=16):
  super().__init__();self.base=base;self.scale=alpha/r
  self.A=nn.Parameter(torch.empty(r,base.in_features,device=base.weight.device,dtype=torch.float32));self.B=nn.Parameter(torch.zeros(base.out_features,r,device=base.weight.device,dtype=torch.float32));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 def forward(self,x):return self.base(x)+(F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(x.dtype)
def setup(adapter_path=None):
 torch.set_num_threads(6)
 cfg=PreTrainedConfig.from_pretrained(str(ROOT/'checkpoint'));cfg.dtype=torch.bfloat16
 policy=XVLAPolicy.from_pretrained(str(ROOT/'checkpoint'),config=cfg).to('cuda').eval();policy.requires_grad_(False)
 targets=[]
 def inject(m,prefix=''):
  for name,child in list(m.named_children()):
   path=prefix+'.'+name
   if isinstance(child,nn.Linear):setattr(m,name,LoRALinear(child));targets.append(path)
   else:inject(child,path)
 inject(policy.model.transformer)
 pre,post=make_pre_post_processors(policy.config,str(ROOT/'checkpoint'),preprocessor_overrides={'device_processor':{'device':'cuda'}},postprocessor_overrides={'device_processor':{'device':'cpu','float_dtype':'float32'}})
 ep,eo=make_xvla_libero_pre_post_processors()
 if adapter_path:
  sd=torch.load(adapter_path,map_location='cpu',weights_only=True);params=dict(policy.named_parameters());assert set(sd)=={n for n,p in params.items() if p.requires_grad}
  with torch.no_grad():
   for n,v in sd.items():params[n].copy_(v)
  policy.requires_grad_(False)
 return policy,pre,post,ep,eo,{'rank':8,'alpha':16,'targets':targets,'trainable_parameters':sum(p.numel() for p in policy.parameters() if p.requires_grad),'domain_id':3,'method':'LoRA only; frozen VLM and all base weights'}
def state(policy):return {n:p.detach().cpu().clone() for n,p in policy.named_parameters() if p.requires_grad}
