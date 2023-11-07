from typing import Dict

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from editor.base import BaseEditor
from util import get_module, get_shape


class MALMEN(BaseEditor):
        
    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):

            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.data.n_edits // self.config.data.batch_size))
            ])
            value_diffs = torch.empty((0, net.value_size), device = self.config.editor_device)
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                with torch.no_grad():
                    pesudo_keys, pesudo_values_grad = net(
                        keys[start_idx:end_idx],
                        values_grad[start_idx:end_idx],
                        layer_idx
                    )
                    coeffs = - net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diffs = torch.cat((value_diffs, coeffs * pesudo_values_grad))
            with torch.no_grad():
                mat = keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.editor_device)
            param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)

        return param_shifts
        
    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor]):
        
        self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size))
            ])
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32).to(self.config.editor_device)
            param_shift = param_shifts[module_name].to(self.config.editor_device)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            with torch.no_grad():
                mat = torch.linalg.solve(keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.editor_device), module_grad)
                lamda_grad = - net.lamda(layer_idx).exp() * (mat * param_shift).sum()
            value_diffs_grad = keys @ mat
            (lamda_grad * net.lamda(layer_idx)).backward()
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                pesudo_keys, pesudo_values_grad = net(
                    keys[start_idx:end_idx],
                    values_grad[start_idx:end_idx],
                    layer_idx
                )
                coeffs = - net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diff = coeffs * pesudo_values_grad
                (value_diffs_grad[start_idx:end_idx] * value_diff).sum().backward()
            
        clip_grad_norm_(
            self.net.parameters(),
            self.config.editor.max_grad_norm
        )
        self.opt.step()  