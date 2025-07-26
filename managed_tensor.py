import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

class ManagedTensor:
    """
    A tensor wrapper that automatically manages device placement for PyTorch operations.
    """
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.device = tensor.device

    def __repr__(self):
        return f"Managed({self.tensor.shape}, device='{self.device}')"

    # --- Properties to mimic a real tensor ---
    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def dtype(self):
        return self.tensor.dtype

    # --- Methods to mimic the tensor API ---
    def sum(self, *args, **kwargs):
        return torch.sum(self, *args, **kwargs)

    def relu(self, *args, **kwargs):
        return F.relu(self, *args, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        target_device = torch.device('cpu')
        highest_cuda_index = -1

        flat_args, _ = torch.utils._pytree.tree_flatten(list(args) + list(kwargs.values()))
        for arg in flat_args:
            if isinstance(arg, ManagedTensor) and arg.device.type == 'cuda':
                if arg.device.index > highest_cuda_index:
                    highest_cuda_index = arg.device.index
                    target_device = arg.device
        
        def move_and_unwrap(x):
            if isinstance(x, ManagedTensor):
                if x.device != target_device:
                    x.tensor = x.tensor.to(target_device)
                    x.device = target_device
                return x.tensor
            return x

        new_args = tree_map(move_and_unwrap, args)
        new_kwargs = tree_map(move_and_unwrap, kwargs)

        raw_output = func(*new_args, **new_kwargs)

        def wrap_output(x):
            if isinstance(x, torch.Tensor):
                return ManagedTensor(x)
            return x
        
        return tree_map(wrap_output, raw_output)

# Create two ManagedTensor objects
a = ManagedTensor(torch.tensor([[1, 2], [3, 4]]))
b = ManagedTensor(torch.tensor([[10, 20], [30, 40]]))