"""Context and query stages of the model."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from .cache import StackCache
from .embedding import GROUP_SIZE, GroupWindows, ValueEmbedding
from .heads import ClassificationHead, OneHotAndLinear, RegressionHead
from .layers import InducedStack, RoleStack, SelfStack
from .normalize import RMSNorm


class ColumnStage(nn.Module):
    """Embed column groups and condition them on Context labels."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        heads: int,
        hidden_width: int,
        num_inducing: int,
        max_classes: int,
        regression_method: Optional[str] = None,
        rank_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_classes = max_classes
        self.value_embedding = ValueEmbedding(
            group_size=GROUP_SIZE,
            embed_dim=embed_dim,
            rank_encoding=rank_encoding,
        )
        self.column_transformer = InducedStack(
            num_layers=num_layers,
            width=embed_dim,
            heads=heads,
            hidden_width=hidden_width,
            num_inducing=num_inducing,
            use_scaling=True,
        )
        if max_classes > 0:
            self.label_embedding: nn.Module = OneHotAndLinear(max_classes, embed_dim)
        elif regression_method == "bar_dist":
            self.label_embedding = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.label_embedding = nn.Linear(1, embed_dim)

    def group_mask(self, values: Tensor) -> Tensor:
        return GroupWindows.group_mask(values)

    def _embed(
        self,
        values: Tensor,
        frequencies: Tensor,
        ranks: Optional[Tensor],
    ) -> Tensor:
        return self.value_embedding.encode(values, frequencies, ranks).transpose(1, 2)

    def _condition(
        self,
        source: Tensor,
        labels: Tensor,
        group_mask: Tensor,
    ) -> Tensor:
        expanded = labels.unsqueeze(1).expand(-1, source.shape[1], -1)
        if self.max_classes > 0:
            embedded = self.label_embedding(expanded.float())
        else:
            embedded = self.label_embedding(expanded.unsqueeze(-1))
        embedded = embedded.to(source.dtype).masked_fill(
            ~group_mask[..., None, None],
            0.0,
        )
        context_size = labels.shape[1]
        return torch.cat(
            (
                source[..., :context_size, :] + embedded,
                source[..., context_size:, :],
            ),
            dim=-2,
        )

    def _label_views(self, labels: Tensor) -> List[Tensor]:
        if self.max_classes <= 0:
            return [labels]
        num_classes = int(labels.max().item()) + 1
        if num_classes <= self.max_classes:
            return [labels]

        digits = math.ceil(math.log(num_classes) / math.log(self.max_classes))
        base = min(math.ceil(num_classes ** (1.0 / digits)), self.max_classes)
        bases = [base] * digits
        product = base**digits
        for index in range(digits):
            if product >= num_classes:
                break
            if bases[index] < self.max_classes:
                product = product // bases[index] * (bases[index] + 1)
                bases[index] += 1

        views = []
        for index in range(len(bases)):
            views.append(
                (labels.long() // math.prod(bases[index + 1 :])) % bases[index]
            )
        return views

    def can_cache(self, labels: Tensor) -> bool:
        if self.max_classes <= 0:
            return True
        return int(labels.max().item()) + 1 <= self.max_classes

    def encode_context(
        self,
        values: Tensor,
        labels: Tensor,
        ranks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[StackCache], Tensor]:
        context_size = labels.shape[1]
        frequencies = self.value_embedding.context_frequencies(
            values[:, :context_size]
        )
        source = self._embed(values, frequencies, ranks)
        mask = self.group_mask(values)
        views = self._label_views(labels)

        if len(views) > 1:
            outputs = [
                self.column_transformer(
                    self._condition(source, view, mask),
                    context_size,
                )
                for view in views
            ]
            output = torch.stack(outputs, dim=0).mean(dim=0)
            cache = None
        else:
            output, cache = self.column_transformer.encode(
                self._condition(source, views[0], mask),
                context_size,
            )
        return output.transpose(1, 2), cache, frequencies

    def decode_query(
        self,
        values: Tensor,
        frequencies: Tensor,
        cache: StackCache,
        ranks: Optional[Tensor] = None,
    ) -> Tensor:
        source = self._embed(values, frequencies, ranks)
        output = self.column_transformer.decode(source, cache)
        return output.transpose(1, 2)

    def forward(
        self,
        values: Tensor,
        labels: Tensor,
        ranks: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encode_context(values, labels, ranks)[0]


class RowStage(nn.Module):
    """Mix column groups within each row using CLS tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        heads: int,
        hidden_width: int,
        num_cls: int,
        readout: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.readout_mode = readout
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.stack = SelfStack(num_layers, embed_dim, heads, hidden_width)
        self.output_norm = RMSNorm(embed_dim)

    def _tokens(self, features: Tensor) -> Tensor:
        batch, rows, _, _ = features.shape
        cls = self.cls_tokens.to(features).expand(
            batch,
            rows,
            self.num_cls,
            self.embed_dim,
        )
        return torch.cat((cls, features), dim=-2)

    def mix(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the CLS prefix and the updated feature tokens."""
        tokens = self._tokens(features)
        output = self.output_norm(self.stack(tokens))
        return output[..., : self.num_cls, :], output[..., self.num_cls :, :]

    def readout(self, features: Tensor) -> Tensor:
        """Read the CLS prefix through a final cross-attention layer."""
        tokens = self._tokens(features)
        output = self.stack.readout(tokens, self.num_cls)
        return self.output_norm(output).flatten(-2)


class RefinementStage(nn.Module):
    """Refine group tokens across rows with Context-side attention."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        heads: int,
        hidden_width: int,
        num_inducing: int,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(embed_dim)
        self.stack = InducedStack(
            num_layers=num_layers,
            width=embed_dim,
            heads=heads,
            hidden_width=hidden_width,
            num_inducing=num_inducing,
            use_scaling=True,
        )

    @staticmethod
    def _to_groups(features: Tensor) -> Tensor:
        return features.transpose(1, 2)

    @staticmethod
    def _to_rows(features: Tensor) -> Tensor:
        return features.transpose(1, 2)

    def forward(self, features: Tensor, context_size: int) -> Tensor:
        grouped = self._to_groups(features)
        output = self.stack(self.input_norm(grouped), context_size)
        return self._to_rows(output)

    def encode_context(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, StackCache]:
        grouped = self._to_groups(features)
        output, cache = self.stack.encode(
            self.input_norm(grouped),
            context_size=grouped.shape[-2],
        )
        return self._to_rows(output), cache

    def decode_query(self, features: Tensor, cache: StackCache) -> Tensor:
        grouped = self._to_groups(features)
        output = self.stack.decode(self.input_norm(grouped), cache)
        return self._to_rows(output)


class ICLStage(nn.Module):
    """In-context learning readout over row representations."""

    def __init__(
        self,
        model_type: str,
        out_dim: int,
        width: int,
        num_layers: int,
        heads: int,
        hidden_width: int,
        max_classes: int,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.max_classes = max_classes
        self.stack = RoleStack(num_layers, width, heads, hidden_width)
        self.boundary_norm = RMSNorm(width, affine=False)
        if model_type == "classifier":
            self.label_embedding: nn.Module = OneHotAndLinear(max_classes, width)
            self.head: nn.Module = ClassificationHead(width, out_dim)
        else:
            self.label_embedding = nn.Sequential(
                nn.Linear(1, width),
                nn.LayerNorm(width),
            )
            self.head = RegressionHead(width, out_dim)

    def _add_labels(self, representations: Tensor, labels: Tensor) -> Tensor:
        context_size = labels.shape[1]
        if self.model_type == "classifier":
            embedded = self.label_embedding(labels.float())
        else:
            embedded = self.label_embedding(labels.unsqueeze(-1))
        embedded = embedded.to(representations.dtype)
        return torch.cat(
            (
                representations[:, :context_size] + embedded,
                representations[:, context_size:],
            ),
            dim=1,
        )

    def encode_context(
        self,
        representations: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, StackCache]:
        full = self._add_labels(representations, labels)
        context, cache = self.stack.encode(full)
        return self.boundary_norm(context), cache

    def decode_query(
        self,
        representations: Tensor,
        cache: StackCache,
        n_classes: int,
    ) -> Tensor:
        query = self.stack.decode(representations, cache)
        logits = self.head(self.boundary_norm(query))
        if self.model_type == "classifier":
            return logits[..., :n_classes]
        return logits

    def predict_standard(
        self,
        representations: Tensor,
        labels: Tensor,
        return_probs: bool = False,
        temperature: float = 1.0,
        n_classes: Optional[int] = None,
    ) -> Tensor:
        context_size = labels.shape[1]
        full = self._add_labels(representations, labels)
        output = self.stack(full, context_size)
        logits = self.head(self.boundary_norm(output[:, context_size:]))
        if self.model_type == "classifier":
            classes = (
                n_classes
                if n_classes is not None
                else int(torch.unique(labels[0]).numel())
            )
            logits = logits[..., :classes]
            if return_probs:
                logits = torch.softmax(logits / temperature, dim=-1)
        return logits

    def prepare_context_rows(
        self,
        representations: Tensor,
        labels: Tensor,
    ) -> Tensor:
        return self._add_labels(representations, labels)[:, : labels.shape[1]]

    def predict_from_context_rows(
        self,
        full_representations: Tensor,
        context_size: int,
        n_classes: int,
    ) -> Tensor:
        output = self.stack(full_representations, context_size)
        logits = self.head(self.boundary_norm(output[:, context_size:]))
        if self.model_type == "classifier":
            return logits[..., :n_classes]
        return logits