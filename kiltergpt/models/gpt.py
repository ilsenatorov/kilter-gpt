import math
from argparse import Namespace

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .. import __version__
from ..data.datasets import climb_signature
from ..data.tokenizer import Tokenizer
from ..utils import Plotter, WarmupCosineSchedule
from ..utils.metrics import generation_metrics, structure_metrics
from ..utils.utils import KilterPolice, validate_prompt
from .grade import GradePredictor

N_FREQS = 6  # Fourier frequencies of the hold positions, 1 to 32 cycles across the board
DEFAULT_CFG_SCALE = 3.0  # with temperature 1 and top-p 1: grade error close to temperature 0.7 without losing holds
# Test-time sampling, temperature -> top-p: the old 0.7 / 0.8 default (comparable with earlier runs) and the new one
TEST_TOP_P = {0.7: 0.8, 1.0: 1.0}


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

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None):
        """bias: optional (B, nh, T, T) additive attention bias, on top of the causal mask"""
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if bias is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            bias = bias.masked_fill(~causal, float("-inf"))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0,
            is_causal=bias is None,
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

    def forward(self, x, bias=None):
        x = x + self.attn(self.ln_1(x), bias)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(L.LightningModule):
    """The non-kilterboard specific GPT model"""

    def __init__(self, config, hold_positions: torch.Tensor | None = None):
        """hold_positions: (vocab_size, 2) board position of each hold token, for config.hold_positions"""
        super().__init__()
        self.config = config
        # Optional: each hold's (tied input/output) embedding adds a function of its board position
        self.position_proj = None
        # Optional: attention between two holds drops with their board distance, at a learned rate per head
        self.distance_slopes = None
        if getattr(config, "hold_positions", False) or getattr(config, "spatial_attention", False):
            self.register_buffer("hold_xy", hold_positions)
            self.register_buffer("is_hold", hold_positions.abs().sum(1, keepdim=True) > 0)
        if getattr(config, "hold_positions", False):
            self.position_proj = nn.Sequential(
                nn.Linear(4 * N_FREQS, config.n_embed), nn.GELU(), nn.Linear(config.n_embed, config.n_embed)
            )
        if getattr(config, "spatial_attention", False):  # half the heads start distance-blind
            self.distance_slopes = nn.Parameter(torch.linspace(-4, 4, config.n_head).clamp(min=0))
        self.angle_proj = nn.Linear(1, self.config.n_embed)
        self.grade_proj = nn.Linear(1, self.config.n_embed)
        # Learned "no grade" embedding for classifier-free guidance; old checkpoints were trained without it
        self.null_grade = nn.Parameter(torch.zeros(config.n_embed)) if getattr(config, "p_uncond", 0) > 0 else None
        # Optional ascent-count input (Tokenizer.popularity) with its own learned "any popularity" embedding
        self.popularity_proj = nn.Linear(1, config.n_embed) if getattr(config, "popularity", False) else None
        self.null_popularity = nn.Parameter(torch.zeros(config.n_embed)) if self.popularity_proj else None

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

    def forward(self, idx, angle, grade, uncond=None, popularity=None):
        x = self.embed(idx, angle, grade, uncond, popularity)
        return F.linear(x, self.token_embeddings())

    def token_embeddings(self) -> torch.Tensor:
        """(vocab_size, n_embed), shared by the input and the output layer"""
        if self.position_proj is None:
            return self.lm_head.weight
        freqs = 2 ** torch.arange(N_FREQS, device=self.hold_xy.device) * math.pi
        angles = (self.hold_xy.unsqueeze(-1) * freqs).flatten(1)  # (vocab, 2 * N_FREQS)
        fourier = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.lm_head.weight + self.is_hold * self.position_proj(fourier)

    def distance_bias(self, idx: torch.Tensor, n_prefix: int) -> torch.Tensor:
        """(B, nh, T, T) attention bias: -slope * board distance between two hold tokens, 0 for any other pair"""
        xy = F.pad(self.hold_xy[idx], (0, 0, n_prefix, 0))  # the prefix and role tokens have no position
        has = F.pad(self.is_hold[idx].squeeze(-1).float(), (n_prefix, 0))
        diff = xy.unsqueeze(2) - xy.unsqueeze(1)
        distance = (diff.pow(2).sum(-1) + 1e-9).sqrt() * has.unsqueeze(2) * has.unsqueeze(1)
        return -self.distance_slopes.view(1, -1, 1, 1) * distance.unsqueeze(1)

    def embed(self, idx, angle, grade, uncond=None, popularity=None):
        """uncond: optional (B,) bool mask replacing the grade with the null embedding
        popularity: optional (B, 1) ascent count on Tokenizer.popularity's scale, for models trained with it;
        None conditions on any popularity. Training drops it to the null embedding at the grade dropout rate."""
        angle = self.angle_proj(angle.unsqueeze(-1))
        grade = self.grade_proj(grade.unsqueeze(-1))
        b, t = idx.size()
        if uncond is None and self.training and self.null_grade is not None:
            uncond = torch.rand(b, device=idx.device) < self.config.p_uncond
        if uncond is not None:
            grade = torch.where(uncond.view(b, 1, 1), self.null_grade, grade)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)
        tok_emb = F.embedding(idx, self.token_embeddings())  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        prefix = [angle, grade]
        if self.popularity_proj is not None:
            null = self.null_popularity.expand(b, 1, -1)
            if popularity is None:
                prefix.append(null)
            else:
                pop = self.popularity_proj(popularity.unsqueeze(-1))
                if self.training:
                    drop = torch.rand(b, device=idx.device) < self.config.p_uncond
                    pop = torch.where(drop.view(b, 1, 1), null, pop)
                prefix.append(pop)
        x = torch.cat((*prefix, x), dim=1)
        bias = None if self.distance_slopes is None else self.distance_bias(idx, len(prefix))
        for block in self.transformer.h:
            x = block(x, bias)
        x = self.transformer.ln_f(x)
        return x


class GPTModel(L.LightningModule):
    """Whole model that is kilterboard-specific"""

    def __init__(self, config, tokenizer):
        super(GPTModel, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        spatial = getattr(config, "hold_positions", False) or getattr(config, "spatial_attention", False)
        positions = tokenizer.hold_positions() if spatial else None
        self.model = GPT(self.config, positions)
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

    def forward(self, x, angle, grade, uncond=None, popularity=None):
        logits = self.model.forward(x, angle, grade, uncond, popularity)
        return logits

    def shared_step(self, batch, name: str):
        # (B, S), (B, 1), (B, 1), (B, S), optionally popularity (B, 1)
        x, angle, grade, y, *popularity = batch
        logits = self.forward(x, angle, grade, popularity=popularity[0] if popularity else None)
        loss = self.get_loss(logits, y)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_test_epoch_start(self) -> None:
        torch.manual_seed(42)
        self.test_generated = {(temp, cfg): [] for temp in TEST_TOP_P for cfg in self.cfg_scales}
        self.test_conditions = []
        self.test_reference = {climb_signature(frames) for frames in self.trainer.datamodule.train.df.frames}
        self.test_police = KilterPolice(self.tokenizer, (1, 2), (1, 2), n_total_holds=(4, 28))

    def test_step(self, batch, batch_idx):
        x, angle, grade, y, *popularity = batch
        popularity = popularity[0] if popularity else None  # generate like each test climb, popularity included
        self.log("test/loss", self.get_loss(self(y[:, :-1], angle, grade, popularity=popularity), y[:, 1:]))
        self.test_conditions.append((round(angle.item() * 70), grade.item() * len(self.tokenizer.grade_tokens())))
        for temp, cfg in self.test_generated:
            generated = self.generate(
                x.squeeze(0),
                angle.squeeze(0),
                grade.squeeze(0),
                temp,
                p=TEST_TOP_P[temp],
                cfg_scale=cfg,
                popularity=None if popularity is None else popularity.squeeze(0),
            )
            self.test_generated[temp, cfg].append(self.tokenizer.decode(generated.detach().cpu(), clean=True))
            self._gen_logits.clear()

    @property
    def cfg_scales(self) -> list[float]:
        return getattr(self.config, "cfg_scales", [1.0])

    def _get_fixed_prompt_diversity(self, temp: float, cfg: float):
        n_runs = 4 if self.config.n_embed < 32 else 16
        climbs = {
            climb_signature(self.generate_from_string("p1387r14", 40, "7a", temp, TEST_TOP_P[temp], cfg))
            for _ in range(n_runs)
        }
        return len(climbs) / n_runs

    def on_test_epoch_end(self) -> None:
        grade_predictor = None
        if getattr(self.config, "grade_predictor", None):
            grade_predictor = GradePredictor.load_checkpoint(self.config.grade_predictor).to(self.device).eval()
        real = [frames for frames in self.trainer.datamodule.test.df.frames]
        self.log_dict({f"test/real/{name}": score for name, score in structure_metrics(real).items()})
        self.log("test/set_nll", self.set_nll(self.trainer.datamodule.test.df))
        for (temp, cfg), generated in self.test_generated.items():
            prefix = f"test/temp={temp}/cfg={cfg}"
            scores = generation_metrics(generated, self.test_reference, self.test_police, real[: len(generated)])
            scores["fixed_prompt_unique_rate"] = self._get_fixed_prompt_diversity(temp, cfg)
            if grade_predictor is not None:
                scores["grade_mae"] = grade_predictor.grade_mae(generated, self.test_conditions, self.test_police)
            self.log_dict({f"{prefix}/{name}": score for name, score in scores.items()})

    @torch.no_grad()
    def set_nll(self, df, n_perms: int = 8, batch_size: int = 512) -> float:
        """Mean negative log-likelihood (nats) of each climb as a set of (hold, role) pairs, given its conditions.
        Comparable across hold orders: a bottom-up model scores its one ordering, a shuffled model the mean
        likelihood of n_perms random orderings times n!. Both are lower bounds on the set likelihood."""
        was_training = self.training
        self.eval()
        bottom_up, n_perms = self.tokenizer.bottom_up, 1 if self.tokenizer.bottom_up else n_perms
        popularity = getattr(self.config, "popularity", False)
        rows = []
        for frames, angle, grade, ascents in df[
            ["frames", "angle", "difficulty_average", "ascensionist_count"]
        ].values:
            for _ in range(n_perms):
                ordered = self.tokenizer.sort_bottom_up(frames) if bottom_up else frames
                x, a, g = self.tokenizer.encode(ordered, int(angle), float(grade), shuffle=not bottom_up)
                rows.append((x, a, g, self.tokenizer.popularity(ascents)))
        lls = []
        try:
            for i in range(0, len(rows), batch_size):
                x, angle, grade, pop = zip(*rows[i : i + batch_size], strict=True)
                x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
                angle, grade = torch.stack(angle).unsqueeze(1).to(self.device), torch.stack(grade).unsqueeze(1)
                pop = torch.tensor(pop).unsqueeze(1).to(self.device) if popularity else None
                logits = self(x[:, :-1], angle, grade.to(self.device), popularity=pop)[:, -x.size(1) + 1 :]
                targets = x[:, 1:]
                token_ll = -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
                lls.append((token_ll * (targets != self.tokenizer.pad_token_id)).sum(1).cpu())
        finally:
            self.train(was_training)
        lls = torch.cat(lls).view(len(df), n_perms)
        n_holds = torch.tensor([frames.count("p") for frames in df["frames"]], dtype=torch.float)
        set_ll = torch.logsumexp(lls, 1) - math.log(n_perms)
        if not bottom_up:
            set_ll = set_ll + torch.lgamma(n_holds + 1)
        return -set_ll.mean().item()

    # TODO add tests
    def on_train_epoch_end(self):
        if self.config.only_train:
            return super().on_train_epoch_end()
        plotter = Plotter()
        finish_hold = "p1387"
        setup = [(30, "6a"), (40, "7a"), (50, "8a")]
        for temp, p in TEST_TOP_P.items():
            route_frames = [
                self.generate_from_string(f"{finish_hold}r14", angle, grade, temperature=temp, p=p)
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
        popularity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a single token
        prompt: torch.LongTensor: input_ids
        cfg_scale: classifier-free guidance weight on the grade, 1.0 is plain conditional sampling
        With a role-first tokenizer, temperature and top-p only shape which hold comes next: roles and the stop
        decision always sample at temperature 1, so lower temperatures keep the foot holds and the climb length.
        """
        tokenizer = self.tokenizer
        role_first = tokenizer.role_first
        last_token = prompt[-1]
        hold_ids = tokenizer.hold_token_ids.to(self.device)
        if cfg_scale == 1.0:
            pop = None if popularity is None else popularity.unsqueeze(0)
            logits = self.forward(prompt.unsqueeze(0), angle.unsqueeze(0), grade.unsqueeze(0), popularity=pop)[0, -1]
        else:
            # conditional and unconditional passes in one batch of 2; only the grade is dropped
            uncond = torch.tensor([False, True], device=self.device)
            pop = None if popularity is None else popularity.expand(2, -1)
            logits = self.forward(prompt.expand(2, -1), angle.expand(2, -1), grade.expand(2, -1), uncond, pop)
            logits = logits[1, -1] + cfg_scale * (logits[0, -1] - logits[1, -1])
        # A climb can stop only between holds: after BOS or a finished (hold, role) pair
        between_holds = last_token == tokenizer.bos_token_id or (
            last_token in hold_ids if role_first else last_token in tokenizer.color_token_ids.to(self.device)
        )
        picks_hold = between_holds != role_first
        used_holds = prompt[torch.isin(prompt, hold_ids)]
        n_holds = len(used_holds)
        n_starts = (prompt == tokenizer.start_token_id).sum().item()
        n_finishes = (prompt == tokenizer.finish_token_id).sum().item()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if picks_hold:
            mask[tokenizer.hold_token_ids] = True
            mask[used_holds] = False
        else:
            mask[tokenizer.color_token_ids] = True
            if n_starts >= 2:
                mask[tokenizer.start_token_id] = False
            if n_finishes >= 2:
                mask[tokenizer.finish_token_id] = False
            # leave room for the start and finish within 28 holds; role first, this role's hold isn't placed yet
            if n_holds + role_first + (n_starts == 0) + (n_finishes == 0) > 28:
                mask[:] = False
                mask[tokenizer.start_token_id if n_starts == 0 else tokenizer.finish_token_id] = True
        if between_holds:
            mask[tokenizer.eos_token_id] = n_holds >= 4 and n_starts > 0 and n_finishes > 0
            if n_holds >= 28:
                mask[:] = False
                mask[tokenizer.eos_token_id] = True
        if role_first and not picks_hold:
            temperature, p = 1.0, 1.0
        logits = logits / temperature
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
        popularity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate until EOS token is reached (not batched). Always samples in eval mode:
        train mode would add dropout and randomly drop the grade (p_uncond)."""
        if cfg_scale != 1.0 and self.model.null_grade is None:
            raise ValueError("cfg_scale != 1 needs a model trained with p_uncond > 0")
        if popularity is not None and self.model.popularity_proj is None:
            raise ValueError("popularity needs a model trained with --popularity")
        was_training = self.training
        self.eval()
        try:
            next_prompt = None
            while next_prompt != self.tokenizer.eos_token_id:
                context = prompt[-self.config.context_len :]
                next_prompt = self._generate_token(
                    context, angle, grade, temperature=temperature, p=p, cfg_scale=cfg_scale, popularity=popularity
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
        ascents: float | int | None = None,
    ) -> str:
        """Generate a climb from a string of frames, angle, and grade.
        Defaults match real climbs: lower temperature or top-p drops foot holds first and cuts climbs short.
        cfg_scale defaults to 3.0 on checkpoints trained with grade dropout and to 1.0 on older ones.
        ascents: for models trained with --popularity, generate climbs like ones with about this many ascents;
        None conditions on any popularity."""
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
        popularity = None
        if ascents is not None:
            popularity = torch.tensor([self.tokenizer.popularity(ascents)], device=self.device)
        generated = self.generate(
            x, angle, grade, temperature=temperature, p=p, cfg_scale=cfg_scale, popularity=popularity
        )
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
        # the hold-choice steps, plus the final stop step (its non-hold tokens are not drawn)
        probs = probs[1::2] + probs[-1:] if self.tokenizer.role_first else probs[0::2]
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
