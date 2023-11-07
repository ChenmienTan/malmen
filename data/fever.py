from typing import Dict

import random

import torch

from data.base import BaseDataset


class FEVERDataset(BaseDataset):

    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]

        prompt = row["prompt"]
        equiv_prompt = random.choice(row["equiv_prompt"])
        unrel_prompt = row["unrel_prompt"]
        alt = row["alt"]
        ans = row["unrel_ans"]

        return {
            "edit_tuples": self.tok_tuples(prompt, alt),
            "equiv_tuples": self.tok_tuples(equiv_prompt, alt),
            "unrel_tuples": self.tok_tuples(unrel_prompt, ans)
        }
    
    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:
        
        tok_tuples = self.tok(
            prompt,
            max_length = 512,
            return_tensors = "pt",
            truncation = True
        )

        tok_tuples["labels"] = torch.FloatTensor([[answer == "SUPPORTS"]])

        return tok_tuples

