"""Attention layers and stacks."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cache import LayerCache, StackCache
from .normalize import RMSNorm
from .ops import attend, swiglu
from .scaling import BoundedQueryScaling


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.expand = nn.Linear(width, 2 * hidden_width)
        self.contract = nn.Linear(hidden_width, width)

    def forward(self, value: Tensor) -> Tensor:
        return self.contract(swiglu(self.expand(value)))


class MultiHeadAttention(nn.Module):
    """Packed-projection multi-head attention."""

    def __init__(
        self,
        width: int,
        heads: int,
        scaling: Optional[BoundedQueryScaling] = None,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads.")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.scaling = scaling

    def _project(self, value: Tensor, start: int, count: int) -> Tensor:
        weight = self.qkv.weight[start * self.width : (start + count) * self.width]
        bias = self.qkv.bias
        if bias is not None:
            bias = bias[start * self.width : (start + count) * self.width]
        return F.linear(value, weight, bias)

    def _heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    @staticmethod
    def _merge(value: Tensor) -> Tensor:
        return value.transpose(-3, -2).flatten(-2)

    def self_attention(self, value: Tensor) -> Tensor:
        query, key, val = self.qkv(value).chunk(3, dim=-1)
        attention = attend(
            self._heads(query),
            self._heads(key),
            self._heads(val),
            self.scaling,
        )
        return self.out(self._merge(attention))

    def project_key_value(self, value: Tensor) -> LayerCache:
        key, val = self._project(value, 1, 2).chunk(2, dim=-1)
        return LayerCache(self._heads(key), self._heads(val))

    def query_attention(self, query: Tensor, cache: LayerCache) -> Tensor:
        query_heads = self._heads(self._project(query, 0, 1))
        attention = attend(
            query_heads,
            cache.keys,
            cache.values,
            self.scaling,
        )
        return self.out(self._merge(attention))


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with self- and cross-attention paths."""

    def __init__(
        self,
        width: int,
        heads: int,
        hidden_width: int,
        scaling: Optional[BoundedQueryScaling] = None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.feedforward_norm = RMSNorm(width)
        self.attention = MultiHeadAttention(width, heads, scaling)
        self.feed_forward = FeedForward(width, hidden_width)

    def _residual(self, state: Tensor, attention_output: Tensor) -> Tensor:
        state = state + attention_output
        return state + self.feed_forward(self.feedforward_norm(state))

    def self_forward(self, value: Tensor) -> Tensor:
        output = self.attention.self_attention(self.attention_norm(value))
        return self._residual(value, output)

    def cross_forward(self, query: Tensor, cache: LayerCache) -> Tensor:
        output = self.attention.query_attention(
            self.attention_norm(query),
            cache,
        )
        return self._residual(query, output)

    def encode_cross(
        self,
        query: Tensor,
        source: Tensor,
    ) -> Tuple[Tensor, LayerCache]:
        cache = self.attention.project_key_value(self.attention_norm(source))
        return self.cross_forward(query, cache), cache


class SelfStack(nn.Module):
    """Stack of self-attention layers."""

    def __init__(
        self,
        num_layers: int,
        width: int,
        heads: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(width, heads, hidden_width)
                for _ in range(num_layers)
            ]
        )

    def forward(self, value: Tensor) -> Tensor:
        output = value
        for layer in self.layers:
            output = layer.self_forward(output)
        return output

    def readout(self, value: Tensor, prefix: int) -> Tensor:
        """Run all but the last layer in self-attention, then read the prefix."""
        output = value
        for layer in self.layers[:-1]:
            output = layer.self_forward(output)
        last = self.layers[-1]
        cache = last.attention.project_key_value(last.attention_norm(output))
        return last.cross_forward(output[..., :prefix, :], cache)


class InducedLayer(nn.Module):
    """Induced-attention layer with separate Context and query paths."""

    def __init__(
        self,
        width: int,
        heads: int,
        hidden_width: int,
        num_inducing: int,
        scaling: Optional[BoundedQueryScaling] = None,
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(num_inducing, width))
        nn.init.trunc_normal_(self.inducing, std=0.02)
        self.context_layer = TransformerLayer(width, heads, hidden_width, scaling)
        self.query_layer = TransformerLayer(width, heads, hidden_width)

    def _inducing(self, source: Tensor) -> Tensor:
        return self.inducing.expand(
            *source.shape[:-2],
            self.inducing.shape[0],
            source.shape[-1],
        )

    def encode(
        self,
        source: Tensor,
        context_size: int,
    ) -> Tuple[Tensor, LayerCache]:
        context = source[..., :context_size, :]
        hidden, _ = self.context_layer.encode_cross(self._inducing(source), context)
        return self.query_layer.encode_cross(source, hidden)

    def decode(self, query: Tensor, cache: LayerCache) -> Tensor:
        return self.query_layer.cross_forward(query, cache)


class InducedStack(nn.Module):
    """Stack of induced-attention layers."""

    def __init__(
        self,
        num_layers: int,
        width: int,
        heads: int,
        hidden_width: int,
        num_inducing: int,
        use_scaling: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                InducedLayer(
                    width,
                    heads,
                    hidden_width,
                    num_inducing,
                    BoundedQueryScaling(heads, width // heads)
                    if use_scaling
                    else None,
                )
                for _ in range(num_layers)
            ]
        )

    def encode(
        self,
        source: Tensor,
        context_size: int,
    ) -> Tuple[Tensor, StackCache]:
        caches = []
        output = source
        for layer in self.layers:
            output, cache = layer.encode(output, context_size)
            caches.append(cache)
        return output, StackCache.from_layers(caches)

    def decode(self, query: Tensor, cache: StackCache) -> Tensor:
        output = query
        for index, layer in enumerate(self.layers):
            output = layer.decode(output, cache[index])
        return output

    def forward(self, source: Tensor, context_size: int) -> Tensor:
        return self.encode(source, context_size)[0]


class RoleLayer(nn.Module):
    """In-context learning layer with role-specific queries and shared cache."""

    def __init__(self, width: int, heads: int, hidden_width: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.kv = nn.Linear(width, 2 * width)
        self.context_q = nn.Linear(width, width)
        self.query_q = nn.Linear(width, width)
        self.context_out = nn.Linear(width, width)
        self.query_out = nn.Linear(width, width)
        self.attention_norm = RMSNorm(width)
        self.feedforward_norm = RMSNorm(width)
        self.feed_forward = FeedForward(width, hidden_width)
        self.query_scaling = BoundedQueryScaling(heads, self.head_dim)

    def _heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    @staticmethod
    def _merge(value: Tensor) -> Tensor:
        return value.transpose(-3, -2).flatten(-2)

    def _consume(
        self,
        state: Tensor,
        cache: LayerCache,
        query_projection: nn.Linear,
        output_projection: nn.Linear,
    ) -> Tensor:
        normalized = self.attention_norm(state)
        query = self._heads(query_projection(normalized))
        attention = attend(
            query,
            cache.keys,
            cache.values,
            self.query_scaling,
        )
        state = state + output_projection(self._merge(attention))
        return state + self.feed_forward(self.feedforward_norm(state))

    def encode(self, context: Tensor) -> Tuple[Tensor, LayerCache]:
        normalized = self.attention_norm(context)
        key, value = self.kv(normalized).chunk(2, dim=-1)
        cache = LayerCache(self._heads(key), self._heads(value))
        output = self._consume(
            context,
            cache,
            self.context_q,
            self.context_out,
        )
        return output, cache

    def decode(self, query: Tensor, cache: LayerCache) -> Tensor:
        return self._consume(query, cache, self.query_q, self.query_out)


class RoleStack(nn.Module):
    """Stack of role-separated in-context learning layers."""

    def __init__(self, num_layers: int, width: int, heads: int, hidden_width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RoleLayer(width, heads, hidden_width)
                for _ in range(num_layers)
            ]
        )

    def encode(self, context: Tensor) -> Tuple[Tensor, StackCache]:
        caches = []
        output = context
        for layer in self.layers:
            output, cache = layer.encode(output)
            caches.append(cache)
        return output, StackCache.from_layers(caches)

    def decode(self, query: Tensor, cache: StackCache) -> Tensor:
        output = query
        for index, layer in enumerate(self.layers):
            output = layer.decode(output, cache[index])
        return output

    def forward(self, source: Tensor, context_size: int) -> Tensor:
        context, query = source[..., :context_size, :], source[..., context_size:, :]
        context, cache = self.encode(context)
        query = self.decode(query, cache)
        return torch.cat((context, query), dim=-2)