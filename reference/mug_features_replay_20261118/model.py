from common import *
def setup():
 torch.set_num_threads(6)
 config=PreTrainedConfig.from_pretrained(str(ROOT/'checkpoint'));config.dtype=torch.bfloat16
 policy=XVLAPolicy.from_pretrained(str(ROOT/'checkpoint'),config=config).to('cuda').eval();policy.requires_grad_(False)
 pre,post=make_pre_post_processors(policy.config,str(ROOT/'checkpoint'),preprocessor_overrides={'device_processor':{'device':'cuda'}},postprocessor_overrides={'device_processor':{'device':'cpu','float_dtype':'float32'}})
 ep,eo=make_xvla_libero_pre_post_processors()
 return policy,pre,post,ep,eo
