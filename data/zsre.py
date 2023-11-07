from typing import Dict

import torch

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from data.base import BaseDataset


class ZSREDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        
        prompt = row["src"]
        equiv_prompt = row["rephrase"]
        answer = row["ans"]
        unrel_prompt = row["loc"] + "?"
        unrel_answer = row["loc_ans"]
    
        return {
            "edit_tuples": self.tok_tuples(prompt, answer),
            "equiv_tuples": self.tok_tuples(equiv_prompt, answer),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }
        
    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:

        if isinstance(self.tok, GPT2TokenizerFast):
            answer = " " + answer
            
        tok_prompt = self.tok(
            prompt,
            return_tensors = "pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors = "pt",
            add_special_tokens = False
        )

        if isinstance(self.tok, GPT2TokenizerFast):

            tok_tuples = {
                key: torch.cat((value, tok_answer[key][:, :-1]), -1)
                for key, value in tok_prompt.items()
            }
            
            tok_tuples["labels"] = torch.cat((
                torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
                tok_answer["input_ids"]
            ), -1)
        
        if isinstance(self.tok, T5TokenizerFast):

            tok_tuples = {
                "input_ids": tok_prompt["input_ids"],
                "attention_mask": tok_prompt["attention_mask"],
                "decoder_input_ids": torch.cat((
                    torch.LongTensor([[0]]),
                    tok_answer["input_ids"][:, :-1]
                ), -1),
                "decoder_attention_mask": tok_answer["attention_mask"],
                "labels": tok_answer["input_ids"]
            }

        return tok_tuples