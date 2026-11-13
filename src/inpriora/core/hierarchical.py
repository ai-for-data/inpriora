"""Hierarchical readout for datasets with many classes."""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch
from torch import Tensor


class ClassNode:
    """Node in a hierarchical class tree."""

    def __init__(self, depth: int = 0) -> None:
        self.depth = depth
        self.is_leaf = False
        self.classes_: Optional[Tensor] = None
        self.children: List["ClassNode"] = []
        self.class_mapping: dict[int, int] = {}
        self.group_indices: Optional[Tensor] = None
        self.representations: Optional[Tensor] = None
        self.labels: Optional[Tensor] = None


class HierarchicalReadout:
    """Balanced class tree that composes per-node predictions."""

    def __init__(self, max_classes: int, num_classes: int | None = None) -> None:
        if max_classes <= 0:
            raise ValueError("max_classes must be positive.")
        self.max_classes = max_classes
        self.num_classes = num_classes
        self.root = ClassNode()

    @staticmethod
    def _label_encoding(labels: Tensor) -> Tensor:
        unique, _ = torch.unique(labels, return_inverse=True)
        order = unique.argsort()
        return order[torch.searchsorted(unique, labels)]

    def _grouping(self, num_classes: int) -> tuple[Tensor, int]:
        groups = min(math.ceil(num_classes / self.max_classes), self.max_classes)
        assignments = torch.zeros(num_classes, dtype=torch.int)
        position = 0
        remaining_classes = num_classes
        remaining_groups = groups
        for group in range(groups):
            size = math.ceil(remaining_classes / remaining_groups)
            assignments[position : position + size] = group
            position += size
            remaining_classes -= size
            remaining_groups -= 1
        return assignments, groups

    def _fit_node(
        self,
        node: ClassNode,
        representations: Tensor,
        labels: Tensor,
        depth: int,
    ) -> None:
        unique = torch.unique(labels).int()
        node.classes_ = unique
        node.representations = representations
        node.labels = labels

        if len(unique) <= self.max_classes:
            node.is_leaf = True
            return

        assignments, groups = self._grouping(len(unique))
        node.class_mapping = {
            int(label.item()): int(group.item())
            for label, group in zip(unique, assignments)
        }
        node.group_indices = torch.tensor(
            [node.class_mapping[int(label.item())] for label in labels],
            dtype=torch.int,
        )
        node.is_leaf = False
        for group in range(groups):
            mask = node.group_indices == group
            child = ClassNode(depth + 1)
            self._fit_node(child, representations[mask], labels[mask], depth + 1)
            node.children.append(child)

    def fit(self, representations: Tensor, labels: Tensor) -> "HierarchicalReadout":
        """Build the class tree from Context representations."""
        self.root = ClassNode()
        self._fit_node(self.root, representations, labels, depth=0)
        return self

    def predict(
        self,
        predict_fn: Callable[..., Tensor],
        representations: Tensor,
        temperature: float,
    ) -> Tensor:
        """Return the full class distribution for query representations."""
        num_classes = (
            self.num_classes
            if self.num_classes is not None
            else int(self.root.classes_.max().item()) + 1
        )
        return self._predict_node(
            self.root,
            representations,
            temperature,
            predict_fn,
            num_classes,
        )

    def _predict_node(
        self,
        node: ClassNode,
        inputs: Tensor,
        temperature: float,
        predict_fn: Callable[..., Tensor],
        num_classes: int,
    ) -> Tensor:
        device = inputs.device
        test_size = inputs.shape[0]
        node_inputs = torch.cat(
            (node.representations.to(device), inputs),
            dim=0,
        )

        if node.is_leaf:
            local_labels = self._label_encoding(node.labels.to(device))
            leaf = predict_fn(
                node_inputs.unsqueeze(0),
                local_labels.unsqueeze(0),
                return_probs=True,
                temperature=temperature,
            ).squeeze(0)
            global_probs = torch.zeros(
                (test_size, num_classes),
                device=device,
            )
            for local_index, global_index in enumerate(node.classes_):
                global_probs[:, global_index] = leaf[:, local_index]
            return global_probs

        group_labels = node.group_indices.to(device)
        group_probs = predict_fn(
            node_inputs.unsqueeze(0),
            group_labels.unsqueeze(0),
            return_probs=True,
            temperature=temperature,
        ).squeeze(0)

        outputs = torch.zeros((test_size, num_classes), device=device)
        for group_index, child in enumerate(node.children):
            child_probs = self._predict_node(
                child,
                inputs,
                temperature,
                predict_fn,
                num_classes,
            )
            outputs += child_probs * group_probs[
                :, group_index : group_index + 1
            ]
        return outputs