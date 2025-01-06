import copy
import inspect
import warnings

import triton

from . import backend
from .backend.device import DeviceDetector


class ConfigLoader(object):
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
        self.yaml_name = yamlname
    
    def update_tuneconfig(self, new_yamlname: str):
        """Updates the autotune configuration file used by the ConfigLoader
        instance.  If the new YAML file name is the same as the current one,
        a warning is triggered.  Otherwise, the current instance is reset,
        and a new instance is created with the new YAML file.

        Args:
            new_yamlname (str): The name of the new autotune configuration
                file to load.

        Returns:
            ConfigLoader: A new instance of ConfigLoader initialized with the
                new YAML file.

        Raises:
            UserWarning: If the new YAML file name is the same as the current
                one, a warning is issued to indicate that no update is needed.

        Example:
            To update the configuration file to 'myBench.yaml', use:
            >>> import flag_gems
            >>> flag_gems.runtime.config_loader.update_tuneconfig("myBench.yaml")
        """
        # if new_yamlname == self.yaml_name:
        if 0:
            # Get the caller information and print the warning message
            caller_frame = inspect.currentframe().f_back
            caller_info = inspect.getframeinfo(caller_frame)
            warning_message = (
                f"Warning: The tune config file '{new_yamlname}' is already loaded. "
                f"No update is needed. \nCalled from {caller_info.filename}, line {caller_info.lineno}."
            )
            # warnings.warn(warning_message, UserWarning)
            return
        else:
            self.yaml_name = new_yamlname
            # Delete the current instance and create a new one
            self.__class__._instance = None  # Reset the singleton instance
            new_instance = self.__class__(new_yamlname)  # Create a new instance
            # warnings.warn(f"{self.__class__._instance}", UserWarning)
            return new_instance

    def __init__(self, yamlname=None):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.device = DeviceDetector()
            # primitive_yaml_config is simply the dictionary returned by yaml
            # and is reserved from being an attr for vendor customizability
            self.yaml_name = None
            if yamlname is None:
                self.primitive_yaml_config = self.get_vendor_tune_config()
            else:
                self.primitive_yaml_config = self.get_vendor_tune_config(yamlname)
                self.yaml_name = yamlname
            self.heuristics_config = self.get_vendor_heuristics_config()
            # gen_key is an identifier that indicates whether the current config needs to be generated automatically
            self.gen_key = "gen"
            # loaded_triton_config is wrapped in triton.Config according to primitive_yaml_config
            self.loaded_triton_config = {}
            self.triton_config_default = {
                "num_stages": 2,
                "num_warps": 4,
                "num_ctas": 1,
            }
            self.load_all()

    def load_all(self):
        for key in self.primitive_yaml_config:
            self.loaded_triton_config[key] = self.get_tuned_config(key)

    def get_vendor_heuristics_config(self):
        return backend.get_heuristic_config(self.device.vendor_name)

    def get_vendor_tune_config(self, yamlname=None):
        return backend.get_tune_config(self.device.vendor_name, yamlname)

    def _gen_impl(
        self,
        gen_config,
        param_config,
        iteration_keys,
        std_config,
    ):
        all_configs = []
        final_step = len(iteration_keys)
        stack = [{"cur_config": std_config, "current_step": 0}]

        while stack:
            cur_state = stack[-1]
            stack.pop()
            cur_config = cur_state.get("cur_config")
            current_step = cur_state.get("current_step")

            if current_step == final_step:
                all_configs.append(
                    triton.Config(
                        cur_config["META"],
                        num_warps=cur_config["num_warps"],
                        num_stages=cur_config["num_stages"],
                        num_ctas=cur_config["num_ctas"],
                    )
                )
            else:
                cur_key = iteration_keys[current_step]
                if cur_key in param_config["META"]:
                    config_var_key = param_config["META"][cur_key]
                else:
                    config_var_key = param_config[cur_key]
                if isinstance(config_var_key, int):
                    key_config = [config_var_key]
                else:
                    key_config = gen_config[config_var_key]
                for single_value in key_config:
                    new_config = copy.deepcopy(cur_config)
                    if cur_key in param_config["META"]:
                        new_config["META"][cur_key] = single_value
                    else:
                        new_config[cur_key] = single_value
                    stack.append(
                        {
                            "cur_config": new_config,
                            "current_step": current_step + 1,
                        }
                    )
        return all_configs

    def to_gen_config(self, gen_config):
        param_config = gen_config["param_map"]
        meta_config = param_config["META"]
        iteration_keys = list(meta_config) + list(param_config)
        iteration_keys.remove("META")
        current_config = {"META": {}}
        current_config.update(self.triton_config_default)
        return self._gen_impl(
            gen_config,
            param_config,
            iteration_keys,
            current_config,
        )

    def get_tuned_config(self, op_name):
        if op_name in self.loaded_triton_config:
            return self.loaded_triton_config[op_name]

        # 
        if not op_name in self.primitive_yaml_config:
            return []
        
        current_op_configs = self.primitive_yaml_config[op_name]
        configs = []
        if len(current_op_configs) == 0:
            return configs

        if len(current_op_configs) == 1:
            single_config = current_op_configs[0]
            if self.gen_key in single_config:
                return self.to_gen_config(single_config)

        for single_config in current_op_configs:
            current_config = self.triton_config_default
            for default_param in current_config:
                if default_param in single_config:
                    current_config[default_param] = single_config[default_param]
            configs.append(
                triton.Config(
                    single_config["META"],
                    num_warps=current_config["num_warps"],
                    num_stages=current_config["num_stages"],
                    num_ctas=current_config["num_ctas"],
                )
            )
        # warnings.warn(f"{configs[0].__str__()}", UserWarning)
        return configs
