import inspect
import math
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from ..utils import Plotter, WarmupCosineSchedule


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

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = F.scaled_dot_product_attention(
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

    def __init__(self, config, tokenizer):
        super(GPTModel, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        # self.model = GPT(self.config)
        self.model = torch.compile(GPT(self.config))

    def get_loss(self, logits, targets):
        B, C, V = logits.shape
        logits = logits.view(B * C, V)
        if len(targets.size()) == 2:  # If targets are class labels
            targets = targets.view(B * C)
            loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.pad_token_id)
        else:  # if targets are class probabilities
            targets = targets.view(B * C, V)
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss

    def forward(self, x):
        logits = self.model.forward(x)
        return logits

    def shared_step(self, batch, name="train"):
        text, target = batch
        logits = self.forward(text)
        loss = self.get_loss(logits, target)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def plot_generated_climbs(self):
        """Used to visually monitor quality of generated data during training"""
        plotter = Plotter()
        for temp in [0.1, 0.2, 0.3, 0.5]:
            texts = [self.generate_from_string("p1133r12", 40, grade, temp) for grade in ["5a", "6a", "7a", "8a"]]
            images = [plotter.plot_climb(x[0]) for x in texts]
            captions = [f"Angle: {x[1]}, Grade: {x[2]}, Temp: {temp}" for x in texts]
            self.logger.log_image(key=f"temp_{temp}", images=images, caption=captions)

    def on_train_epoch_end(self):
        if self.current_epoch % 25 == 0 and self.current_epoch > 0:
            self.plot_generated_climbs()

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.wd},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(0.9, 0.95), fused=True)
        scheduler = WarmupCosineSchedule(
            optimizer,
            self.config.total_steps // 10,
            self.config.total_steps,
            0.01,
            0.1,
        )
        lr_scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def embed(self, x: torch.Tensor):
        """Embeds prompts of size (B, C) into (B, C, Z) where Z is the embedding dimension"""
        return self.model.embed(x)

    def _generate_token(self, prompts: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """Generate a single token"""
        logits = self.forward(prompts)
        logits = logits[:, -1, :] / temperature
        logit_probs = F.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        return next_prompt

    def generate(self, prompt: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """Generate until EOS token is reached (not batched)"""
        assert prompt.size() == (1, self.config.context_len), "Prompt shape must be (1, context_len)"
        # Generate until EOS token is reached
        for _ in range(999):  # this could be a while loop, but just to be safe I use a for loop
            context = prompt[:, -self.config.context_len :]
            next_prompt = self._generate_token(context, temperature)
            prompt = torch.cat((prompt, next_prompt), dim=1)
            if next_prompt == self.tokenizer.eos_token_id:
                break
        return prompt

    def generate_batch(self, prompts: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """Generate until EOS token is reached (batched)"""
        for _ in range(999):
            context = prompts[:, -self.config.context_len :]
            next_prompt = self._generate_token(context, temperature)
            prompt = torch.cat([prompt, next_prompt], dim=1)
            # if eos token is present in every sample, break
            if (prompt == self.tokenizer.eos_token_id).any(dim=1).all():
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
        generated = self.generate(tokenized.unsqueeze(0), temperature)
        return self.tokenizer.decode(generated.squeeze(0), clean=True)
