import os
from dataclasses import dataclass
from enum import Enum

import yaml
import warnings

class Autograd(Enum):
    enable = True
    disable = False

    @classmethod
    def get_optional_value(cls):
        return [member.name for member in cls]


# Metadata template,  Each vendor needs to specialize instances of this template
@dataclass
class VendorInfoBase:
    vendor_name: str
    device_name: str
    device_query_cmd: str


def get_tune_config(vendor_name, yamlname, file_mode="r"):
    try:
        vendor_name = "_" + vendor_name
        script_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(script_path)
        if yamlname is None:
            file_path = os.path.join(base_dir, vendor_name, "tune_configs.yaml")
            # warnings.warn(f"{file_path}", UserWarning)
        else:
            file_path = os.path.join(base_dir, vendor_name, yamlname)
            # warnings.warn(f"{file_path}", UserWarning)
        with open(file_path, file_mode) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    return config

    # def update_tuneconfig(self, new_yamlname: str):
    #     """Updates the autotune configuration file used by the ConfigLoader instance.
    #     If the new YAML file name is given, the current instance is reset, and a new
    #     instance is created with the new YAML file.

    #     Args:
    #         new_yamlname (str): The name of the new autotune configuration
    #             file to load.

    #     Returns:
    #         ConfigLoader: A new instance of ConfigLoader initialized with the
    #             new YAML file.

    #     Example:
    #         To update the configuration file to 'myBench.yaml', use:
    #         >>> import flag_gems
    #         >>> flag_gems.runtime.config_loader.update_tuneconfig("myBench.yaml")
    #     """
    #     # Delete the current instance and create a new one
    #     self.__class__._instance = None
    #     new_instance = self.__class__(new_yamlname)
    #     return new_instance