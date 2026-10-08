import math
from argparse import Namespace

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import __version__
from ..data.datasets import climb_signature
from ..data.tokenizer import Tokenizer
from ..utils import Plotter, WarmupCosineSchedule
from ..utils.metrics import generation_metrics
from ..utils.utils import KilterPolice, validate_prompt
from .grade import GradePredictor

DEFAULT_CFG_SCALE = 3.0  # with temperature 1 and top-p 1: grade error close to temperature 0.7 without losing holds


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
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
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
        self.angle_proj = nn.Linear(1, self.config.n_embed)
        self.grade_proj = nn.Linear(1, self.config.n_embed)
        # Learned "no grade" embedding for classifier-free guidance; old checkpoints were trained without it
        self.null_grade = nn.Parameter(torch.zeros(config.n_embed)) if getattr(config, "p_uncond", 0) > 0 else None

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

    def forward(self, idx, angle, grade, uncond=None):
        x = self.embed(idx, angle, grade, uncond)
        logits = self.lm_head(x)
        return logits

    def embed(self, idx, angle, grade, uncond=None):
        """uncond: optional (B,) bool mask replacing the grade with the null embedding"""
        angle = self.angle_proj(angle.unsqueeze(-1))
        grade = self.grade_proj(grade.unsqueeze(-1))
        b, t = idx.size()
        if uncond is None and self.training and self.null_grade is not None:
            uncond = torch.rand(b, device=idx.device) < self.config.p_uncond
        if uncond is not None:
            grade = torch.where(uncond.view(b, 1, 1), self.null_grade, grade)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = torch.cat((angle, grade, x), dim=1)
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
        self._gen_logits = []

    # TODO add tests
    def get_loss(self, logits, targets):
        # batch, sequence, vocab
        if len(targets.size()) == 2:  # Only 2 dimensions -> class labels (batch, label)
            B, S = targets.size()
            logits = logits[:, -S:, :]
            targets = targets.reshape(B * S)
            logits = logits.reshape(B * S, -1)
            loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.pad_token_id)
        elif len(targets.size()) == 3:  # multilabel classification (batch, sequence, vocab)
            B, S, V = targets.size()
            logits = logits[:, -S:, :]
            targets = targets.reshape(B * S, V)
            logits = logits.reshape(B * S, V)
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss

    def forward(self, x, angle, grade, uncond=None):
        logits = self.model.forward(x, angle, grade, uncond)
        return logits

    def shared_step(self, batch, name: str):
        # (B, S), (B, 1), (B, 1), (B, S)
        x, angle, grade, y = batch
        logits = self.forward(x, angle, grade)
        loss = self.get_loss(logits, y)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_test_epoch_start(self) -> None:
        torch.manual_seed(42)
        self.test_generated = {(temp, cfg): [] for temp in (0.3, 0.5, 0.7) for cfg in self.cfg_scales}
        self.test_conditions = []
        self.test_reference = {climb_signature(frames) for frames in self.trainer.datamodule.train.df.frames}
        self.test_police = KilterPolice(self.tokenizer, (1, 2), (1, 2), n_total_holds=(4, 28))

    def test_step(self, batch, batch_idx):
        x, angle, grade, y = batch
        self.log("test/loss", self.get_loss(self(y[:, :-1], angle, grade), y[:, 1:]))
        self.test_conditions.append((round(angle.item() * 70), grade.item() * len(self.tokenizer.grade_tokens())))
        for temp, cfg in self.test_generated:
            generated = self.generate(x.squeeze(0), angle.squeeze(0), grade.squeeze(0), temp, p=0.8, cfg_scale=cfg)
            self.test_generated[temp, cfg].append(self.tokenizer.decode(generated.detach().cpu(), clean=True))
            self._gen_logits.clear()

    @property
    def cfg_scales(self) -> list[float]:
        return getattr(self.config, "cfg_scales", [1.0])

    def _get_fixed_prompt_diversity(self, temp: float, cfg: float):
        n_runs = 4 if self.config.n_embed < 32 else 16
        climbs = {
            climb_signature(self.generate_from_string("p1387r14", 40, "7a", temp, 0.8, cfg)) for _ in range(n_runs)
        }
        return len(climbs) / n_runs

    def on_test_epoch_end(self) -> None:
        grade_predictor = None
        if getattr(self.config, "grade_predictor", None):
            grade_predictor = GradePredictor.load_checkpoint(self.config.grade_predictor).to(self.device).eval()
        for (temp, cfg), generated in self.test_generated.items():
            prefix = f"test/temp={temp}/cfg={cfg}"
            scores = generation_metrics(generated, self.test_reference, self.test_police)
            scores["fixed_prompt_unique_rate"] = self._get_fixed_prompt_diversity(temp, cfg)
            if grade_predictor is not None:
                scores["grade_mae"] = grade_predictor.grade_mae(generated, self.test_conditions, self.test_police)
            self.log_dict({f"{prefix}/{name}": score for name, score in scores.items()})

    # TODO add tests
    def on_train_epoch_end(self):
        if self.config.only_train:
            return super().on_train_epoch_end()
        plotter = Plotter()
        finish_hold = "p1387"
        setup = [(30, "6a"), (40, "7a"), (50, "8a")]
        for temp in [0.3, 0.5, 0.7]:
            route_frames = [
                self.generate_from_string(f"{finish_hold}r14", angle, grade, temperature=temp, p=0.8)
                for angle, grade in setup
            ]
            route_images = [plotter.plot_climb(x, highlight=finish_hold) for x in route_frames]
            captions = [f"{grade} @ {angle}, temp={temp}" for angle, grade in setup]
            self.logger.log_image(key=f"test/temp={temp}/images", images=route_images, caption=captions)
        return super().on_train_epoch_end()

    # TODO add tests
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.wd,
            betas=(0.9, 0.95),
        )
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

    def _sample_from_logits(self, logits: torch.Tensor, p: float) -> torch.Tensor:
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
        self._gen_logits.append(probs)
        next_prompt = torch.multinomial(probs, num_samples=1).to(self.device)
        return next_prompt

    def _generate_token(
        self,
        prompt: torch.Tensor,
        angle: torch.Tensor,
        grade: torch.Tensor,
        temperature: float,
        p: float,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate a single token
        prompt: torch.LongTensor: input_ids
        cfg_scale: classifier-free guidance weight on the grade, 1.0 is plain conditional sampling
        """
        last_token = prompt[-1]
        if cfg_scale == 1.0:
            logits = self.forward(prompt.unsqueeze(0), angle.unsqueeze(0), grade.unsqueeze(0))[0, -1]
        else:
            # conditional and unconditional passes in one batch of 2
            uncond = torch.tensor([False, True], device=self.device)
            logits = self.forward(prompt.expand(2, -1), angle.expand(2, -1), grade.expand(2, -1), uncond)[:, -1]
            logits = logits[1] + cfg_scale * (logits[0] - logits[1])
        logits = logits / temperature
        # After color/BOS token only hold or EOS token can be generated
        if last_token in self.tokenizer.color_token_ids.to(self.device) or last_token == self.tokenizer.bos_token_id:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            # Only allow hold tokens and EOS token
            mask[self.tokenizer.hold_token_ids] = True
            mask[self.tokenizer.eos_token_id] = True
            previously_used = prompt[torch.isin(prompt, self.tokenizer.hold_token_ids.to(self.device))]
            mask[previously_used] = False
            n_holds = len(previously_used)
            if (
                n_holds < 4
                or not (prompt == self.tokenizer.start_token_id).any()
                or not (prompt == self.tokenizer.finish_token_id).any()
            ):
                mask[self.tokenizer.eos_token_id] = False
            if n_holds >= 28:
                mask[:] = False
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
            missing_start = not (prompt == self.tokenizer.start_token_id).any()
            missing_finish = not (prompt == self.tokenizer.finish_token_id).any()
            n_holds = torch.isin(prompt, self.tokenizer.hold_token_ids.to(self.device)).sum().item()
            if n_holds + missing_start + missing_finish > 28:
                mask[:] = False
                mask[self.tokenizer.start_token_id if missing_start else self.tokenizer.finish_token_id] = True
            logits[~mask] = float("-inf")
        return self._sample_from_logits(logits, p)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        angle: torch.Tensor,
        grade: torch.Tensor,
        temperature: float,
        p: float,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate until EOS token is reached (not batched). Always samples in eval mode:
        train mode would add dropout and randomly drop the grade (p_uncond)."""
        if cfg_scale != 1.0 and self.model.null_grade is None:
            raise ValueError("cfg_scale != 1 needs a model trained with p_uncond > 0")
        was_training = self.training
        self.eval()
        try:
            next_prompt = None
            while next_prompt != self.tokenizer.eos_token_id:
                context = prompt[-self.config.context_len :]
                next_prompt = self._generate_token(
                    context, angle, grade, temperature=temperature, p=p, cfg_scale=cfg_scale
                )
                prompt = torch.cat((prompt, next_prompt), dim=0)
                # Stop when you get to full context window (30 holds)
                if prompt.size(0) >= self.config.context_len - 1:
                    prompt = torch.cat(
                        (prompt, torch.tensor(self.tokenizer.eos_token_id, device=self.device).unsqueeze(0)),
                        dim=0,
                    )
                    break
            return prompt
        finally:
            self.train(was_training)

    @torch.jit.export
    def generate_from_string(
        self,
        frames: str,
        angle: int,
        grade: str | float,
        temperature: float = 1.0,
        p: float = 1.0,
        cfg_scale: float | None = None,
    ) -> str:
        """Generate a climb from a string of frames, angle, and grade.
        Defaults match real climbs: lower temperature or top-p drops foot holds first and cuts climbs short.
        cfg_scale defaults to 3.0 on checkpoints trained with grade dropout and to 1.0 on older ones."""
        if cfg_scale is None:
            cfg_scale = DEFAULT_CFG_SCALE if self.model.null_grade is not None else 1.0
        validate_prompt(frames)
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(p) or not 0 < p <= 1:
            raise ValueError("Temperature must be positive and top-p must be in (0, 1]")
        if not math.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError("cfg_scale must be a non-negative number")
        self._gen_logits = []
        x, angle, grade = self.tokenizer.encode(frames, angle, grade, eos=False)
        x, angle, grade = x.to(self.device), angle.to(self.device).unsqueeze(0), grade.to(self.device).unsqueeze(0)
        generated = self.generate(x, angle, grade, temperature=temperature, p=p, cfg_scale=cfg_scale)
        return self.tokenizer.decode(generated, clean=True)

    def generate_generation_fig(
        self,
        prompt: str,
        angle: int,
        grade: str,
        temperature: float = 1.0,
        p: float = 1.0,
        name: str | None = None,
        colorscheme: str = "RdPu",
        cutoff: float = 0.1,
    ):
        import imageio

        plotter = Plotter(colorscheme, verbose_cutoff=cutoff)
        frames = self.generate_from_string(prompt, angle, grade, temperature, p)
        probs = self._gen_logits
        probs = probs[0::2]
        split_frames = [f"p{i}" for i in frames.split("p")[1:]]
        split_frames = ["".join(split_frames[:i]) for i in range(len(split_frames))]
        split_frames.append(frames)
        animation = []
        for frame, prob in zip(split_frames, probs, strict=True):
            dict_prob = {}
            for i in torch.nonzero(prob):
                dict_prob[self.tokenizer.decode_map[i.item()]] = prob[i].item()
            animation.append((frame, dict_prob))
        images = []
        for i in range(len(animation)):
            k, v = animation[i]
            # plot the current frame + options
            images.append(plotter.plot_climb(k, probs=v))
            if i < len(animation) - 1:
                # plot the made choice
                next_k, _ = animation[i + 1]
                images.append(plotter.plot_climb(next_k, probs=v))
        images.append(plotter.plot_climb(frames))
        if name is None:
            name = f"{prompt}_{angle}_{grade}_{temperature}_{p}.gif"
        imageio.mimsave(name, images, fps=1)
        return frames

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> "GPTModel":
        with torch.serialization.safe_globals([Namespace, Tokenizer]):
            return GPTModel.load_from_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)

    @staticmethod
    def load_from_wandb(checkpoint_path: str) -> "GPTModel":
        """Use self.load_from_checkpoint to download model weights from wandb. Looks for models in ilsenatorov/kilter-gpt"""
        import wandb

        api = wandb.Api()
        artifact = api.artifact(checkpoint_path)
        artifact_dir = artifact.download()
        return GPTModel.load_checkpoint(f"{artifact_dir}/model.ckpt")
