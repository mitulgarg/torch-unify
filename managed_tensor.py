import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map


class ManagedTensor:
    def __init__(self, tensor: torch.Tensor, device: str = None):
        """
        Initializes the ManagedTensor.

        If a specific `device` is not provided, it will automatically move the
        tensor to 'cuda' if a GPU is available, otherwise leaving it on the CPU.
        """
        # --- NEW: Smart Constructor Logic ---
        if device is None:
            # Automatic placement: Use GPU if available
            if torch.cuda.is_available():
                target_device = torch.device('cuda')
                self.tensor = tensor.to(target_device)
                self.device = target_device
            else:
                # Fallback to CPU
                self.tensor = tensor
                self.device = tensor.device
        else:
            # Manual override: Respect the user's choice
            target_device = torch.device(device)
            self.tensor = tensor.to(target_device)
            self.device = target_device

   
    def __repr__(self):
        return f"Managed({self.tensor.shape}, device='{self.device}')"

    @property
    def shape(self): 
        return self.tensor.shape
    
    @property
    def dtype(self): 
        return self.tensor.dtype
    
    def __getattr__(self, name): 
        return getattr(self.tensor, name)
    
    def sum(self, *args, **kwargs): 
        return torch.sum(self, *args, **kwargs)
    
    def relu(self, *args, **kwargs): 
        return F.relu(self, *args, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # --- Rule: Priority is cuda:0 > cuda:1 > ... > cpu ---
        target_device = torch.device('cpu')
        highest_cuda_index = -1

        flat_args, _ = torch.utils._pytree.tree_flatten(list(args) + list(kwargs.values()))
        for arg in flat_args:
            if isinstance(arg, ManagedTensor) and arg.device.type == 'cuda':
                if arg.device.index > highest_cuda_index:
                    highest_cuda_index = arg.device.index
                    target_device = arg.device
        
        # --- Move Tensors and Unwrap ---
        def move_and_unwrap(x):
            if isinstance(x, ManagedTensor):
                if x.device != target_device:
                    x.tensor = x.tensor.to(target_device)
                    x.device = target_device
                return x.tensor
            # Also handle regular tensors that might be part of the operation
            if isinstance(x, torch.Tensor):
                return x.to(target_device)
            return x

        new_args = tree_map(move_and_unwrap, args)
        new_kwargs = tree_map(move_and_unwrap, kwargs)

        # --- Execute the original function ---
        raw_output = func(*new_args, **new_kwargs)

        # --- Wrap the output back into a ManagedTensor ---
        def wrap_output(x):
            if isinstance(x, torch.Tensor):
                return ManagedTensor(x)
            return x
        
        return tree_map(wrap_output, raw_output)
