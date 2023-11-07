from omegaconf import DictConfig

import torch
import torch.nn as nn

import transformers
from transformers import AutoModel

from util import get_module

class AutoModelForFEVER(nn.Module):

    def __init__(self, name_or_path: str):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(name_or_path)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, **kwargs):

        hidden_states = self.backbone(**{
            k: v for k, v in kwargs.items() if k != "labels"
        })["last_hidden_state"][:, 0]
        logits = self.classifier(hidden_states)
        
        return {"logits": logits}


def make_model(config: DictConfig):
    
    if config.class_name == "AutoModelForFEVER":
        model = AutoModelForFEVER(config.name_or_path)
        model.load_state_dict(torch.load(config.weight_path))
    else:
        model_class = getattr(transformers, config.class_name)
        model = model_class.from_pretrained(config.name_or_path)

    if config.half:
        model.bfloat16()

    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.edit_modules:
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model