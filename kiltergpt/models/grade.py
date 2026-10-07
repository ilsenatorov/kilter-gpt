from argparse import Namespace

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..data.tokenizer import Tokenizer
from ..utils import WarmupCosineSchedule


class GradePredictor(L.LightningModule):
    """Predicts difficulty (kilter difficulty units, 16 = 6a/V3) from holds + angle.
    Holds are an unordered set: hold and role embeddings are summed per hold, no positional embedding,
    and the angle token acts as the [CLS] token for the regression head."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = Tokenizer()
        self.n_grades = len(self.tokenizer.grade_tokens())
        self.emb = nn.Embedding(self.tokenizer.vocab_size, config.n_embed, padding_idx=self.tokenizer.pad_token_id)
        self.angle_proj = nn.Linear(1, config.n_embed)
        layer = nn.TransformerEncoderLayer(
            config.n_embed, config.n_head, 4 * config.n_embed, config.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_layer, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(config.n_embed), nn.Linear(config.n_embed, 1))

    def forward(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """x: (B, S) tokens [BOS, hold, role, hold, role, ...], angle: (B, 1) normalized.
        Returns (B,) normalized grade, same scale as the dataset's grade."""
        holds, roles = x[:, 1::2], x[:, 2::2]
        tokens = torch.cat((self.angle_proj(angle).unsqueeze(1), self.emb(holds) + self.emb(roles)), dim=1)
        pad = torch.cat((torch.zeros_like(holds[:, :1], dtype=torch.bool), holds == self.tokenizer.pad_token_id), 1)
        return self.head(self.encoder(tokens, src_key_padding_mask=pad)[:, 0]).squeeze(-1)

    def shared_step(self, batch, name: str):
        x, angle, grade, _ = batch
        mae = F.l1_loss(self(x, angle), grade.squeeze(-1))
        self.log(f"{name}/mae", mae * self.n_grades, batch_size=x.size(0))  # in difficulty units
        return mae

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = WarmupCosineSchedule(optimizer, self.config.total_steps // 20, self.config.total_steps, 0.01, 0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def predict(self, frames: list[str], angles: list[int], batch_size: int = 512) -> torch.Tensor:
        """Predicted difficulty for each climb, in difficulty units"""
        preds = []
        for i in range(0, len(frames), batch_size):
            chunk = zip(frames[i : i + batch_size], angles[i : i + batch_size], strict=True)
            encoded = [self.tokenizer.encode(f, a, eos=False) for f, a in chunk]
            x = pad_sequence([t for t, _, _ in encoded], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            angle = torch.stack([a for _, a, _ in encoded]).unsqueeze(1)
            preds.append(self(x.to(self.device), angle.to(self.device)).cpu())
        return torch.cat(preds) * self.n_grades

    def grade_mae(self, generated: list[str], conditions: list[tuple[int, float]], police) -> float:
        """Mean |predicted - requested| difficulty over the valid generated climbs.
        conditions: (angle, requested difficulty) per generated climb"""
        pairs = [(frames, cond) for frames, cond in zip(generated, conditions, strict=True) if police.check(frames)]
        if not pairs:
            return float("nan")
        frames, conditions = zip(*pairs, strict=True)
        angles, requested = zip(*conditions, strict=True)
        return (self.predict(list(frames), list(angles)) - torch.tensor(requested)).abs().mean().item()

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> "GradePredictor":
        with torch.serialization.safe_globals([Namespace]):
            return GradePredictor.load_from_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)
