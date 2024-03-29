import importlib
from typing import Any

import torch
from collections import OrderedDict

# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)

def load_pytorch_model(ckpt_name, model):
    state_dict = torch.load(ckpt_name)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("model."):
            name = name.replace("model.", "")  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def custom_load(ckpt_pth):
    state_dict = torch.load(ckpt_pth)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict['encoder.'+name] = v
    return new_state_dict
