from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets import MALMENNet

from tqdm import tqdm
import wandb

from util import (
    get_module,
    get_shape,
    empty_cache,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios
)


class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model
        
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1

        self.net = nn.ModuleDict({
            str(k): MALMENNet(
                *k,
                config.editor.rank,
                config.editor.n_blocks,
                v,
                config.editor.lr
            )
            for k, v in shape_counter.items()
        }).to(config.editor_device)
        
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            config.editor.meta_lr
        )
        if config.editor.load_checkpoint:
            self.net.load_state_dict(torch.load(f"checkpoints/{config.model.name_or_path}_{config.editor.name}_{str(config.data.n_edits)}_net.pth"))
            self.opt.load_state_dict(torch.load(f"checkpoints/{config.model.name_or_path}_{config.editor.name}_{str(config.data.n_edits)}_opt.pth"))

    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)

    def train(self, loader: DataLoader):
        
        for tuples in tqdm(loader, desc = "Train", ncols = 100):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.model.zero_grad()

            gen_losses = []
            self.edit_model(param_shifts, False)
            for t in tuples["equiv_tuples"]:
                logits = self.model(**t)["logits"]
                loss = cross_entropy(logits, t["labels"])
                loss.backward()
                gen_losses += [loss.item()]
            self.edit_model(param_shifts, True)

            loc_losses = []
            for t in tuples["unrel_tuples"]:
                with torch.no_grad():
                    refer_logits = self.model(**t)["logits"]

                self.edit_model(param_shifts, False)
                logits = self.model(**t)["logits"]
                loss = kl_div(
                    refer_logits,
                    logits,
                    t["labels"]
                )
                (self.config.editor.loc_coef * loss).backward()
                self.edit_model(param_shifts, True)
                loc_losses += [loss.item()]
                
            self.update_hypernet(param_shifts)

            wandb.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })
    
    def valid(self, loader: DataLoader):
          
        for tuples in tqdm(loader, desc = "Valid", ncols = 100):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, False)
            edit_succs, gen_succs, loc_succs = [], [], []
            for k, s in zip(
                ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                [edit_succs, gen_succs, loc_succs]
            ):
                for t in tuples[k]:
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    s += succ_ratios(logits, t["labels"])
                    
            self.edit_model(param_shifts, True)
            
            wandb.log({
                "ES": np.mean(edit_succs),
                "GS": np.mean(gen_succs),
                "LS": np.mean(loc_succs)
            })
    
    def cache(self, tuples: List[Dict[str, torch.LongTensor]]):

        for idx, t in enumerate(tuples):
            
            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(**t)["logits"]
                cross_entropy(logits, t["labels"]).backward()
        
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                torch.save(keys, f"{self.config.editor.cache_dir}/{module_idx}_{idx}_keys.pth")
                torch.save(values_grad, f"{self.config.editor.cache_dir}/{module_idx}_{idx}_values_grad.pth")
    
    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        
        empty_cache(self.config.editor.cache_dir)
        for _ in range(self.config.editor.n_epochs):
            self.train(train_loader)
            self.valid(valid_loader)
        
        torch.save(self.net.state_dict(), f"checkpoints/{self.config.model.name_or_path}_{self.config.editor.name}_{str(self.config.data.n_edits)}_net.pth")
        torch.save(self.opt.state_dict(), f"checkpoints/{self.config.model.name_or_path}_{self.config.editor.name}_{str(self.config.data.n_edits)}_opt.pth")