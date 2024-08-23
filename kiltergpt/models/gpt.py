import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as M
from fastapi import FastAPI

from ..utils import Plotter, WarmupCosineSchedule
from ..utils.metrics import get_histogram, jaccard_similarity


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
        self.model = GPT(self.config)

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

    def shared_step(self, batch, name: str):
        text, target = batch
        logits = self.forward(text)
        loss = self.get_loss(logits, target)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_test_epoch_start(self) -> None:
        self.test_generated = {0.1: [], 0.3: [], 0.5: [], 0.7: []}
        self.test_real = {0.1: [], 0.3: [], 0.5: [], 0.7: []}
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        prompts, targets = batch
        for temp in self.test_generated.keys():
            for prompt, target in zip(prompts, targets, strict=True):
                generated = self.generate(prompt, temp, p=0.8)
                generated = generated[generated != self.tokenizer.pad_token_id]
                target = target[target != self.tokenizer.pad_token_id]
                self.test_generated[temp].append(generated.detach().cpu())
                self.test_real[temp].append(target.detach().cpu())

    def _get_hist_pearson(self, generated, real):
        """Calculate the spearman correlation between token distributions of real and genenerated sequences"""
        hist_generated = get_histogram(generated, self.tokenizer.vocab_size)
        hist_real = get_histogram(real, self.tokenizer.vocab_size)
        hist_spearman = M.spearman_corrcoef(hist_generated, hist_real)
        return hist_spearman

    def _get_jaccard_similarity(self, generated, real):
        generated_onehot = torch.stack([self.tokenizer.onehot(x) for x in generated])
        real_onehot = torch.stack([self.tokenizer.onehot(x) for x in real])
        jaccard = jaccard_similarity(generated_onehot, real_onehot)
        return jaccard

    def _get_num_possible_climbs(self, temp: float):
        n_runs = 40
        # test run, if the model is too small
        if self.config.n_embed < 32:
            n_runs = 4
        climbs = set()
        for _ in range(n_runs):
            climb = self.generate_from_string("p1387r14", 40, "7a", temp, 0.7)
            onehot = self.tokenizer.onehot(climb)
            climbs.add(tuple(onehot.tolist()))
        return len(climbs) / n_runs

    def on_test_epoch_end(self):
        for temp in self.test_generated.keys():
            prefix = f"test/temp={temp}"
            hist_spearman = self._get_hist_pearson(self.test_generated[temp], self.test_real[temp])
            jaccard_similarity = self._get_jaccard_similarity(self.test_generated[temp], self.test_real[temp])
            num_possible_climbs = self._get_num_possible_climbs(temp)
            self.log_dict(
                {
                    f"{prefix}/hist_spearman": hist_spearman,
                    f"{prefix}/jaccard_similarity": jaccard_similarity.mean(),
                    f"{prefix}/num_possible_climbs": num_possible_climbs,
                }
            )

    def on_train_epoch_end(self):
        plotter = Plotter()
        finish_hold = "p1387"
        setup = [(30, "6a"), (40, "7a"), (50, "8a")]
        for temp in [0.1, 0.3, 0.5, 0.7]:
            route_frames = [
                self.generate_from_string(f"{finish_hold}r14", angle, grade, temperature=temp, p=0.8)
                for angle, grade in setup
            ]
            route_images = [plotter.plot_climb(x, highlight=finish_hold) for x in route_frames]
            captions = [f"{grade} @ {angle}, temp={temp}" for angle, grade in setup]
            self.logger.log_image(key=f"test/temp={temp}/images", images=route_images, caption=captions)
        return super().on_train_epoch_end()

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

    def _sample_from_logits(self, logits: torch.Tensor, p: float = 1.0) -> torch.Tensor:
        """Given logits of tokens, sample using top-p sampling"""
        if p < 1.0:
            # sort by probability, get cumulative probs
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(probs, num_samples=1).to(self.device)
        return next_prompt

    def _generate_token(self, prompt: torch.Tensor, temperature: float = 0.2, p: float = 1.0) -> torch.Tensor:
        """Generate a single token
        prompt: torch.LongTensor: A left-padded tensor of token ids
        """
        # TODO additionally check the number of start and finish tokens
        last_token = prompt[-1]
        logits = self.forward(prompt.unsqueeze(0)).squeeze(0)
        logits = logits[-1, :] / temperature  # Get logits for the last position
        # After color a token only hold or EOS token can be generated
        if last_token in self.tokenizer.color_token_ids.to(self.device):
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[self.tokenizer.hold_token_ids] = True
            mask[self.tokenizer.eos_token_id] = True
            logits[~mask] = float("-inf")
        # After a hold token only color token can be generated
        elif last_token in self.tokenizer.hold_token_ids.to(self.device):
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[self.tokenizer.color_token_ids] = True
            # not more than 2 starts
            # TODO check that this works
            if (prompt == self.tokenizer.start_token_id).to(torch.long).sum() >= 2:
                mask[self.tokenizer.start_token_id] = False
            # not more than 2 finishes
            if (prompt == self.tokenizer.finish_token_id).to(torch.long).sum() >= 2:
                mask[self.tokenizer.finish_token_id] = False
            # All not allowed tokens set prob to 0
            logits[~mask] = float("-inf")
        return self._sample_from_logits(logits, p)

    def generate(self, prompt: torch.Tensor, temperature: float = 0.2, p: float = 1.0) -> torch.Tensor:
        """Generate until EOS token is reached (not batched)"""
        next_prompt = None
        while next_prompt != self.tokenizer.eos_token_id:
            context = prompt[-self.config.context_len :]
            next_prompt = self._generate_token(context, temperature, p)
            prompt = torch.cat((prompt, next_prompt), dim=0)
            # Stop when you get to full context window (30 holds)
            if prompt[prompt != self.tokenizer.pad_token_id].size(0) >= self.config.context_len - 1:
                prompt = torch.cat(
                    (prompt, torch.tensor(self.tokenizer.eos_token_id, device=self.device).unsqueeze(0)), dim=0
                )
                break
        return prompt

    def generate_from_string(
        self,
        frames: str,
        angle: int,
        grade: str,
        temperature: float = 0.2,
        p: float = 0.8,
    ) -> str:
        """Generate a climb from a string of frames, angle, and grade"""
        tokenized = self.tokenizer.encode(frames, angle, grade, pad=self.config.context_len, eos=False).to(self.device)
        generated = self.generate(tokenized, temperature, p)
        return self.tokenizer.decode(generated, clean=True)[0]

    @staticmethod
    def load_from_wandb(wandb_model_name: str) -> "GPTModel":
        """Use self.load_from_checkpoint to download model weights from wandb. Looks for models in ilsenatorov/kilter-gpt"""
        import os

        file_path = f"artifacts/{wandb_model_name}/model.ckpt"
        if not os.path.exists(file_path):
            import wandb

            api = wandb.Api()
            artifact = api.artifact(f"ilsenatorov/kilter-gpt/{wandb_model_name}")
            artifact.download()
        return GPTModel.load_from_checkpoint(file_path)

    def get_fastapi_app(self) -> FastAPI:
        """Return a FastAPI app that serves the model. Can be launched with gunicorn."""
        from fastapi import FastAPI

        app = FastAPI()
        self.eval()
        self.to("cpu")

        @app.get("/generate")
        def generate(frames: str, angle: int, grade: str, temperature: float = 0.2, p: float = 1.0):
            with torch.no_grad():
                result = self.generate_from_string(frames, angle, grade, temperature, p)
            return {"climb": result}

        return app
