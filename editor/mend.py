from typing import Dict

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from editor.base import BaseEditor
from util import get_module, get_shape


class MEND(BaseEditor):

    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            param_shift = torch.zeros((net.key_size, net.value_size), device = self.config.editor_device)
            for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_keys.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_values_grad.pth")
                with torch.no_grad():
                    pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                    param_shift += - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
            param_shifts[module_name] = param_shift

        return param_shifts
        
    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor]):
        
        self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.model.edit_modules,):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_keys.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_values_grad.pth")
                pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                param_shift = - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
                (module_grad * param_shift).sum().backward()
            
        clip_grad_norm_(
            self.net.parameters(),
            self.config.editor.max_grad_norm
        )
        self.opt.step()