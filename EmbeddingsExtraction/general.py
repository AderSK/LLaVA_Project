import sys
import torch
from torch.nn.parameter import is_lazy
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def test_gpu_availability(cfg=None):
    print(f"Using torch {torch.__version__}", file=sys.stdout)
    print(f"Cuda available: {torch.cuda.is_available()}", file=sys.stdout)
    print('__CUDNN VERSION:', torch.backends.cudnn.version(), file=sys.stdout)
    print('Available devices ', torch.cuda.device_count(), file=sys.stdout)
    print('Current cuda device ', torch.cuda.current_device(), file=sys.stdout)
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if cfg and cfg.training.device == "gpu" and torch.cuda.is_available():
        print("Using GPU", file=sys.stdout)
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Using CPU", file=sys.stdout)
        device = torch.device("cpu")

    return device

def check_versions(expected_cuda="12.4", expected_python="3.10"):
    # Check Python version
    current_py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_py_version != expected_python:
        print(f"⚠️ Warning: Expected Python {expected_python}, but found Python {current_py_version}.")

    # Check CUDA version (if available)
    cuda_version = torch.version.cuda
    if cuda_version is None:
        print("⚠️ Warning: CUDA not available in PyTorch installation.")
    elif not cuda_version.startswith(expected_cuda):
        print(f"⚠️ Warning: Expected CUDA {expected_cuda}, but found CUDA {cuda_version}.")


def printable_params(cls):
    """ Decorator to add a method to print trainable parameters of a class."""
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad and not is_lazy(p))
        total = sum(p.numel() for p in self.parameters() if not is_lazy(p))
        print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/total*100:.4f}")
    cls.print_trainable_parameters = print_trainable_parameters
    return cls

def batch_of_dicts_to_outer_dict(batch):
    if (not isinstance(batch, list)) or len(batch) == 0:
        return batch

    outer = {}
    first_dictionary = batch[0]

    if not isinstance(first_dictionary, dict):
        return batch

    for k in first_dictionary.keys():
        outer[k] = [dictionary[k] for dictionary in batch]
        if torch.is_tensor(outer[k][0]):
            outer[k] = pad_sequence(outer[k], batch_first=True)
    return outer
