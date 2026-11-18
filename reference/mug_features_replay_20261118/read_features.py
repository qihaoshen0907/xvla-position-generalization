"""Losslessly decode tensors, including BF16 stored as uint16 bit patterns."""
import numpy as np
import torch

def tensor(dataset):
 a=dataset[()]
 t=torch.from_numpy(np.asarray(a))
 if dataset.attrs.get('torch_dtype')=='torch.bfloat16':t=t.view(torch.bfloat16)
 return t

def array(dataset):
 t=tensor(dataset)
 return t.float().numpy() if t.dtype==torch.bfloat16 else t.numpy()
