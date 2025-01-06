from . import backend, commom_utils
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

# config_loader = ConfigLoader(yamlname="tune_configs.yaml")
config_loader = ConfigLoader()
device = DeviceDetector()

"""
The dependency order of the sub-directory is strict, and changing the order arbitrarily may cause errors.
"""

# torch_device_fn is like 'torch.cuda' object
backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()

# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


def get_tuned_config(op_name):
    config_loader.update_tuneconfig("tune_configs.yaml")
    res = config_loader.get_tuned_config(op_name)
    import warnings
    if len(res) > 0:
        warnings.warn(f"warning using config: {config_loader.get_tuned_config(op_name)[0].__str__()}", UserWarning)
    # else:
    #     warnings.warn(f"yyy {config_loader.get_tuned_config(op_name).__str__()}", UserWarning)
    return config_loader.get_tuned_config(op_name)


def get_heuristic_config(op_name):
    return config_loader.heuristics_config[op_name]


__all__ = [
    "commom_utils",
    "backend",
    "device",
    "get_tuned_config",
    "get_heuristic_config",
]
