import math
import os
from argparse import Namespace

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as metrics

from ..utils import Plotter, Tokenizer


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embed
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(L.LightningModule):
    """The non-kilterboard specific GPT model"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.context_len, config.n_embed),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embed, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)
        logits = self.lm_head(x)
        return logits

    def embed(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x


class GPTModel(L.LightningModule):
    """Whole model that is kilterboard-specific"""

    def __init__(self, config, tokenizer: Tokenizer):
        super(GPTModel, self).__init__()
        # Model Architecture
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.model = GPT(self.config)

    def get_loss(self, logits, targets):
        B, C, V = logits.shape
        logits = logits.view(B * C, V)
        if len(targets.size()) == 2:  # If targets are class labels
            targets = targets.view(B * C)
            mask = targets != self.tokenizer.pad_token_id
            loss = nn.functional.cross_entropy(logits[mask], targets[mask])
        else:  # if targets are class probabilities
            targets = targets.view(B * C, V)
            mask = targets[:, self.tokenizer.pad_token_id] != 1
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])
        return loss

    def forward(self, x):
        logits = self.model(x)
        return logits

    def shared_step(self, batch, name="train"):
        text, target = batch
        logits = self(text)
        loss = self.get_loss(logits, target)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def generate_from_prompts(self):
        if os.path.exists("data/prompts.pt"):
            prompts = torch.load("data/prompts.pt")
            plotter = Plotter()
            tokenized_prompts = torch.stack(
                [self.tokenizer.encode(*x, pad=self.config.context_len, eos=False) for x in prompts]
            ).to(self.device)
            for temp in [0.01, 0.1, 0.3, 0.5]:
                generated = self.generate_batch(tokenized_prompts, 70, temperature=temp)
                texts = [self.tokenizer.decode(x, clean=True) for x in generated]
                images = [plotter.plot_climb(x[0]) for x in texts]
                captions = [f"Angle: {x[1]}, Grade: {x[2]}, Temp: {temp}" for x in texts]
                self.logger.log_image(key=f"temp_{temp}", images=images, caption=captions)

    def on_train_epoch_end(self):
        if self.current_epoch % 25 == 0:
            self.generate_from_prompts()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)

        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                verbose=True,
            ),
            "interval": "epoch",
            "monitor": "val/loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def embed(self, prompts: torch.Tensor):
        """Embeds prompts of size (B, C) into (B, C, Z) where Z is the embedding dimension"""
        return self.model.embed(prompts)

    def generate_batch(self, prompts: torch.Tensor, max_tokens: int, temperature: float = 0.2) -> torch.Tensor:
        """Generates climbs from a batch of tokenized and padded prompts"""
        assert prompts.size(1) == self.config.context_len, "Prompts shape must match context length"
        context = prompts
        for _ in range(max_tokens):
            context = prompts[:, -self.config.context_len :]
            logits = self.forward(context)  # (batch, context_len, vocab_size)
            logits = logits[:, -1, :] / temperature
            logit_probs = nn.functional.softmax(logits, dim=-1)
            next_prompt = torch.multinomial(logit_probs, num_samples=1)
            prompts = torch.cat((prompts, next_prompt), dim=1)
        return prompts

    def generate_single(self, prompt: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """Generate until EOS token is reached (not batched)"""
        assert prompt.size() == (1, self.config.context_len), "Prompt shape must be (1, context_len)"
        context_len = prompt.shape[1]
        context = prompt
        # Generate until EOS token is reached
        for _ in range(999):
            context = prompt[:, -context_len:]
            logits = self.forward(context)
            logits = logits[:, -1, :] / temperature
            logit_probs = nn.functional.softmax(logits, dim=-1)
            next_prompt = torch.multinomial(logit_probs, num_samples=1)
            prompt = torch.cat((prompt, next_prompt), dim=1)
            if next_prompt == self.tokenizer.eos_token_id:
                break
        return prompt

    def generate_from_string(
        self,
        frames: str,
        angle: int,
        grade: str,
        temperature: float = 0.2,
    ) -> tuple[str, str, str]:
        """Generate a climb from a string of frames, angle, and grade"""
        tokenized = self.tokenizer.encode(frames, angle, grade, pad=self.config.context_len, eos=False).to(self.device)
        generated = self.generate_single(tokenized.unsqueeze(0), temperature)
        return self.tokenizer.decode(generated.squeeze(0), clean=True)
