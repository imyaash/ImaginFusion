import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd as bwd, custom_fwd as fwd

class TruncExp(Function):
    """
    Custom autograd Function for the truncated exponential operation.

    This function computes the exponential of input values while clamping the output to a maximum value of 15.

    Args:
        ctx (Context): A PyTorch context object to save intermediate values for backpropagation.
        x (Tensor): The input tensor to compute the truncated exponential for.

    Returns:
        Tensor: The tensor containing the truncated exponential of the input values.
    """
    @staticmethod
    @fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @bwd
    def backward(ctx, g):
        """
        Backward pass for the truncated exponential operation.

        This function computes the gradient of the truncated exponential operation.

        Args:
            ctx (Context): A PyTorch context object containing saved intermediate values.
            g (Tensor): The gradient tensor backpropagated from the next layer.

        Returns:
            Tensor: The gradient tensor with respect to the input values.
        """
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max = 15))

# Creating an alias for the TruncExp custom function
truncExp = TruncExp.apply

def softplus(x, bias = 0):
    """
    Compute the softplus activation function.

    The softplus function is defined as softplus(x) = ln(1 + exp(x)), and this implementation
    allows an optional bias to be applied before computing the softplus.

    Args:
        x (Tensor): The input tensor to apply the softplus activation to.
        bias (float): An optional bias value to be subtracted from the input tensor before applying softplus.

    Returns:
        Tensor: The tensor containing the softplus activations.
    """
    return torch.nn.functional.softplus(x - bias)