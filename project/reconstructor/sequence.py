# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, cast

from esm.modules import (
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
)

from esm.axial_attention import RowSelfAttention, ColumnSelfAttention

import torch.nn.functional as F
import numpy as np
import copy


@dataclass
class CNNModelConfig:
    """Configuration for CNNModel."""

    hidden_dim: int
    num_cnn_stacks: int
    dropout: float = 0.0
    clean_data: bool = False
    cls_expanded_simplex: bool = False
    mode: Literal["dirichlet", "riemannian", "ardm", "lrar", "distill"] = "dirichlet"
    cls_free_guidance: bool = False

    @classmethod
    def from_args(cls, args) -> "CNNModelConfig":
        """
        Create a CNNModelConfig from an argparse args object or dict.

        This is useful for backward compatibility when loading checkpoints
        that were saved with the old args-based approach.

        Args:
            args: Either an argparse Namespace or a dict with the required fields.

        Returns:
            A properly typed CNNModelConfig instance.

        Example:
            >>> # Loading from a checkpoint's hparams
            >>> import yaml
            >>> with open('hparams.yaml') as f:
            >>>     hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            >>> config = CNNModelConfig.from_args(hparams['args'])
            >>> model = CNNModel(config, alphabet_size=4, num_cls=3)
        """

        # Handle both dict and Namespace
        def get_value(key: str, default):
            if isinstance(args, dict):
                return args.get(key, default)
            else:
                return getattr(args, key, default)

        return cls(
            hidden_dim=int(get_value("hidden_dim", 128)),
            num_cnn_stacks=int(get_value("num_cnn_stacks", 1)),
            dropout=float(get_value("dropout", 0.0)),
            clean_data=bool(get_value("clean_data", False)),
            cls_expanded_simplex=bool(get_value("cls_expanded_simplex", False)),
            mode=cast(
                Literal["dirichlet", "riemannian", "ardm", "lrar", "distill"],
                get_value("mode", "dirichlet"),
            ),
            cls_free_guidance=bool(get_value("cls_free_guidance", False)),
        )


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CNNModel(nn.Module):
    def __init__(
        self,
        config: CNNModelConfig,
        alphabet_size: int,
        num_cls: int,
        classifier: bool = False,
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.config = config
        self.classifier = classifier
        self.num_cls = num_cls

        if self.config.clean_data:
            self.linear = nn.Embedding(
                self.alphabet_size, embedding_dim=config.hidden_dim
            )
        else:
            expanded_simplex_input = (
                config.cls_expanded_simplex
                or not classifier
                and (config.mode == "dirichlet" or config.mode == "riemannian")
            )
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            if (config.mode == "ardm" or config.mode == "lrar") and not classifier:
                inp_size += 1  # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, config.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(
                GaussianFourierProjection(embed_dim=config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )

        self.num_layers = 5 * config.num_cnn_stacks
        self.convs = [
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=9,
                dilation=4,
                padding=16,
            ),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=9,
                dilation=16,
                padding=64,
            ),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=9,
                dilation=64,
                padding=256,
            ),
        ]
        self.convs = nn.ModuleList(
            [
                copy.deepcopy(layer)
                for layer in self.convs
                for i in range(config.num_cnn_stacks)
            ]
        )
        self.time_layers = nn.ModuleList(
            [Dense(config.hidden_dim, config.hidden_dim) for _ in range(self.num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(self.num_layers)]
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim if classifier else self.alphabet_size,
                kernel_size=1,
            ),
        )
        self.dropout = nn.Dropout(config.dropout)
        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, self.num_cls),
            )

        if self.config.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(
                num_embeddings=self.num_cls + 1, embedding_dim=config.hidden_dim
            )
            self.cls_layers = nn.ModuleList(
                [
                    Dense(config.hidden_dim, config.hidden_dim)
                    for _ in range(self.num_layers)
                ]
            )

    def forward(self, seq, t, cls=None, return_embedding=False):
        if self.config.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.config.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.config.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.config.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat


class MSATransformer(nn.Module):
    @classmethod
    def add_args(cls, parser):
        # fmt: off
        parser.add_argument(
            "--num_layers",
            default=12,
            type=int,
            metavar="N",
            help="number of layers"
        )
        parser.add_argument(
            "--embed_dim",
            default=768,
            type=int,
            metavar="N",
            help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias",
            action="store_true",
            help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=3072,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=12,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--attention_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--activation_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--max_tokens_per_msa",
            default=2 ** 14,
            type=int,
            help=(
                "Used during inference to batch attention computations in a single "
                "forward pass. This allows increased input sizes with less memory."
            ),
        )
        # fmt: on

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )

        if getattr(self.args, "embed_positions_msa", False):
            emb_dim = getattr(self.args, "embed_positions_msa_dim", self.args.embed_dim)
            self.msa_position_embedding = nn.Parameter(
                0.01 * torch.randn(1, 1024, 1, emb_dim),
                requires_grad=True,
            )
        else:
            self.register_parameter("msa_position_embedding", None)

        self.dropout_module = nn.Dropout(self.args.dropout)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    self.args.dropout,
                    self.args.attention_dropout,
                    self.args.activation_dropout,
                    getattr(self.args, "max_tokens_per_msa", self.args.max_tokens),
                )
                for _ in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions,
            self.args.embed_dim,
            self.padding_idx,
        )
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens)
        x += self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(
            x.size()
        )
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x += self.msa_position_embedding[:, :num_alignments]

        x = self.emb_layer_norm_before(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            row_attn_weights = []
            col_attn_weights = []

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                x, col_attn, row_attn = x
                # H x C x B x R x R -> B x H x C x R x R
                col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
                # H x B x C x C -> B x H x C x C
                row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # col_attentions: B x L x H x C x R x R
            col_attentions = torch.stack(col_attn_weights, 1)
            # row_attentions: B x L x H x C x C
            row_attentions = torch.stack(row_attn_weights, 1)
            result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if return_contacts:
                contacts = self.contact_head(tokens, row_attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers

    def max_tokens_per_msa_(self, value: int) -> None:
        """The MSA Transformer automatically batches attention computations when
        gradients are disabled to allow you to pass in larger MSAs at test time than
        you can fit in GPU memory. By default this occurs when more than 2^14 tokens
        are passed in the input MSA. You can set this value to infinity to disable
        this behavior.
        """
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value
