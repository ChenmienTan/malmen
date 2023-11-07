from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig

import math
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer

class BaseDataset(Dataset):

    def __init__(
        self,
        config: DictConfig,
        path: str,
        tok: AutoTokenizer,
        device: Union[int, str, torch.device]
    ):

        self.config = config
        with open(path) as file:
            self.data = json.load(file)
        self.tok = tok
        self.device = device  

    def __len__(self):
        return len(self.data)

    def collate_fn(
        self,
        tuples: Tuple[Dict[str, Dict[str, torch.LongTensor]]]
    ) -> Dict[str, List[Dict[str, torch.LongTensor]]]:
        
        tuples: Dict[str, List[Dict[str, torch.LongTensor]]] = {
            k: sorted(
                [t[k] for t in tuples],
                key = lambda x: x["attention_mask"].sum().item(),
                reverse = True
            )
            for k in tuples[0].keys()
        }
        
        return {
            k: [
                self.pad_tok_tuples(v[n_batch * self.config.batch_size:(n_batch + 1) * self.config.batch_size])
                for n_batch in range(math.ceil(self.config.n_edits / self.config.batch_size))
            ]
            for k, v in tuples.items()
        }
        
    def pad_tok_tuples(
        self,
        tok_tuples: List[Dict[str, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        
        return {
            k: pad_sequence(
                [t[k].squeeze(0) for t in tok_tuples],
                batch_first = True,
                padding_value = -100 if k == "labels" else 0
            ).to(self.device)
            for k in tok_tuples[0].keys()
        }
    

def make_loader(
    config: DictConfig,
    data_class
) -> Tuple[DataLoader]:
    
    tok = AutoTokenizer.from_pretrained(config.model.name_or_path)

    train_set = data_class(
        config.data,
        config.data.train_path,
        tok,
        config.model_device
    )

    valid_set = data_class(
        config.data,
        config.data.valid_path,
        tok,
        config.model_device
    )

    train_loader = DataLoader(
        train_set,
        config.data.n_edits,
        True,
        collate_fn = train_set.collate_fn,
        drop_last = True
    )

    valid_loader = DataLoader(
        valid_set,
        config.data.n_edits,
        True,
        collate_fn = valid_set.collate_fn,
        drop_last = True
    )

    return train_loader, valid_loader