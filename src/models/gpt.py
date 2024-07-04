import warnings

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from ..utils import EncoderDecoder, Tokenizer


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttentionHead, self).__init__()
        self.config = config

        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.attn_drop = nn.Dropout(config.attn_drop_value)
        self.register_buffer("tril", torch.tril(torch.ones(config.context_len, config.context_len)))

    def forward(self, x):
        # x.shape: (Batch, Context Length, Embedding Dimension)
        B, C, N = x.shape
        q = self.query(x)  # (B, C, head_size)
        k = self.key(x)  # (B, C, head_size)
        v = self.value(x)  # (B, C, head_size)

        # Compute Attention scores
        # (B, C, head_size) bmm (B, head_size, C) -> (B, C, C)
        attn_weight = torch.div(torch.bmm(q, k.permute(0, 2, 1)), self.config.head_size)
        attn_weight = attn_weight.masked_fill(self.tril[:C, :C] == 0, float("-inf"))
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_drop(attn_weight)

        # Do weighted aggregation of values
        output = torch.bmm(attn_weight, v)
        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.embed_dim = config.n_embed

        self.heads = nn.ModuleList([CausalSelfAttentionHead(config) for _ in range(self.num_heads)])
        self.proj = nn.Linear(config.num_heads * config.head_size, config.n_embed)
        self.drop = nn.Dropout(config.multihead_drop_value)

    def forward(self, x):
        multihead_output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.drop(self.proj(multihead_output))


class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.GELU(),
            nn.Linear(config.n_embed * 4, config.n_embed),
            nn.Dropout(config.ffn_drop_value),
        )

    def forward(self, x):
        return self.ffn(x)


class GPTBlock(nn.Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        self.multiheaded_attn = MultiHeadedAttention(config)
        self.ffn = FFN(config)
        self.layernorm1 = nn.LayerNorm(config.n_embed)
        self.layernorm2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.layernorm1(self.multiheaded_attn(x))
        x = x + self.layernorm2(self.ffn(x))
        return x


class GPT(L.LightningModule):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.save_hyperparameters()
        self.config = config
        # Init layers and stuff
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = nn.Embedding(config.context_len, config.n_embed)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.num_blocks)])
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x):
        # Input is just tokenized text of 'B' batches, each 'C' context length long
        B, C = x.shape
        # First we apply the token embedding -> tok_emb (B, C, V)
        tok_emb = self.tok_embedding(x)
        # Then we get the positional embeddings with length equal to context len
        pos_emb = self.pos_embedding(torch.arange(C, device=self.device))
        # Then we add them
        x = tok_emb + pos_emb
        # Then we pass the input through all the GPT blocks
        x = self.blocks(x)
        # And finally pass it through the final layer to get the logits
        logits = self.lm_head(x)
        return logits


class GPTModel(L.LightningModule):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        # Model Architecture
        self.config = config
        self.model = GPT(self.config)

    def get_loss(self, logits, targets):
        B, C, V = logits.shape
        logits = logits.view(B * C, V)
        targets = targets.view(B * C)
        loss = nn.functional.cross_entropy(logits, targets)
        return loss

    def forward(self, x):
        logits = self.model(x)
        return logits

    def shared_step(self, batch, name="train"):
        text, target = batch
        text = text.long()
        target = target.long()
        logits = self(text)
        loss = self.get_loss(logits, target)
        self.log(f"{name}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def generate_from_prompts(self):
        prompts = torch.load("prompts.pt").to(self.device)
        encdec = EncoderDecoder()
        tokenizer = Tokenizer(pd.read_csv("data/raw/gpt_subset.csv")["frames"])
        for temp in [0.1, 0.5, 0.7, 0.9]:
            generated = self.generate(prompts, 64, temperature=temp)
            text = tokenizer.decode_batch(generated)
            images = []
            for t in text:
                t = t.split("[EOS]")[0].split("[BOS]")[-1].replace(" ", "").replace("[PAD]", "").replace("[UNK]", "")
                images.append(encdec.plot_climb(t))
            self.logger.log_image(key=f"temp_{temp}", images=images)

    def on_train_epoch_end(self):
        if self.current_epoch % 25 == 0:
            self.generate_from_prompts()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        return [opt], []

    def generate(self, prompts, max_tokens, temperature=0.7):
        """
        Generates text based on the provided prompts.
        Model determinism can be changed with temperature (range: [0, 1], higher means more unstable but creative predictions)
        """
        self.eval()
        context = prompts
        for _ in range(max_tokens):
            context = prompts[:, -self.config.context_len :]
            logits = self.forward(context)  # (batch, context_len, vocab_size)
            logits = logits[:, -1, :] / temperature
            logit_probs = nn.functional.softmax(logits, dim=-1)
            next_prompt = torch.multinomial(logit_probs, num_samples=1)
            prompts = torch.cat((prompts, next_prompt), dim=1)
        return prompts
