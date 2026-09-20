#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:11:27 2023

@author: zhangxu
"""
import os
import torch.nn as nn
import torch    
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

class LanguageModel(torch.nn.Module):
    """
    Wrapper for loading and running inference with a causal language model.
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device
        )

        # Configure padding for batch inference (left padding for decoder-only models)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        contents: str | list[str],
        max_input_tokens: int = 512,
        max_new_tokens: int = 128
    ):
        """
        Generate outputs for a batch of input prompts.

        Args:
            contents: str or list[str]
            max_input_tokens: maximum length for input truncation
            max_new_tokens: maximum number of tokens to generate

        Returns:
            List[str]: decoded outputs
        """

        # =========================
        # Normalize input to list
        # =========================
        if isinstance(contents, str):
            contents = [contents]

        # =========================
        # Build chat-style messages
        # =========================
        messages = [
            [{"role": "user", "content": c}]
            for c in contents
        ]

        # =========================
        # Apply chat template (model-specific formatting)
        # =========================
        text_batch = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # =========================
        # Tokenization
        # =========================
        inputs = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_input_tokens,
        ).to(self.model.device)

        # =========================
        # Generation
        # =========================
        outputs = self.model.generate( # type: ignore
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic decoding
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # =========================
        # Remove prompt tokens from generated output
        # =========================
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

        # =========================
        # Decode tokens into text
        # =========================
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return decoded

class Model(nn.Module):   
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder

        for name ,param in self.encoder.named_parameters():
            for ele in args.frozen_layers:
                if ele in name:
                    param.requires_grad = False
                    break

    def forward(self, code_inputs=None, nl_inputs=None): 
                
        if code_inputs is not None:
    
            output1 = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (output1*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

        else:
            output2 = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (output2*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

        
class CoModel(nn.Module):   
    def __init__(self, encoder, args):
        super(CoModel, self).__init__()
        self.encoder = encoder
        self.args = args

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.poly_m = args.poly_m
        self.poly_code_embeddings = nn.Embedding(self.poly_m, args.poly_code_dim).to(self.args.device)  
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        #torch.nn.init.normal_(self.poly_code_embeddings.weight, 768 ** -0.5)

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        #print("attn_weights",attn_weights.shape)

        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output
    
    def encode(self, code_inputs=None, nl_inputs=None):
        code_hidden, nl_hidden = None, None

        if code_inputs is not None:
            code_hidden = self.encoder(
                code_inputs,
                attention_mask=code_inputs.ne(1)
            )[0]

        if nl_inputs is not None:
            nl_hidden = self.encoder(
                nl_inputs,
                attention_mask=nl_inputs.ne(1)
            )[0]

        return code_hidden, nl_hidden

    def cross(self, ctx_out, cand_out):
        bs = ctx_out.size(0)

        poly_code_ids = torch.arange(self.poly_m, device=self.args.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(bs, self.poly_m)

        poly_codes = self.poly_code_embeddings(poly_code_ids)

        # poly attention over context
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out)

        # candidate uses CLS
        cand_emb = cand_out[:, 0, :].unsqueeze(1)

        ctx_emb = self.dot_attention(cand_emb, embs, embs)

        return ctx_emb[:, 0, :]

    def forward(self, code_inputs=None, nl_inputs=None):

        # ===== encode ONCE ONLY =====
        code_hidden, nl_hidden = self.encode(code_inputs, nl_inputs)

        v1, v2 = None, None

        # code -> nl
        if code_hidden is not None and nl_hidden is not None:
            v1 = self.cross(code_hidden, nl_hidden)
            v2 = self.cross(nl_hidden, code_hidden)

        return v1, v2
    