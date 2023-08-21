import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd as bwd, custom_fwd as fwd

class TruncatedExponentialFunction(Function):
    @staticmethod
    @fwd(cast_inputs = torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)
    @staticmethod
    @bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g  * torch.exp(x.clamp(max = 15))

truncExp = TruncatedExponentialFunction.apply

def softplus(x, bias = 0):
    return F.softplus(x - bias)