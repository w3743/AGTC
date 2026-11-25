"""Shared local-neuron recurrent classifier for the MNIST first experiment.

The graph has 49 fixed 4x4-average-pooled pixel patch sensors, a sparse hidden
recurrent population, and ten readout cells. Every synapse is processed by the
same edge rule using only its sender's scalar signal, the receiving cell's
state, and that synapse's own weight/history. Cell updates use one shared rule.
Fast edge state is reset for each image batch and evolves only across ticks.
"""
from __future__ import annotations

import random
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EDGE_INPUT = 0
EDGE_RECURRENT = 1
EDGE_READOUT = 2
_EDGE_NAMES = {
    EDGE_INPUT: "input",
    EDGE_RECURRENT: "recurrent",
    EDGE_READOUT: "readout",
}
_NAME_TO_EDGE = {name: key for key, name in _EDGE_NAMES.items()}


class LocalNeuronClassifier(nn.Module):
    """A small shared-rule network with independent state on each edge.

    The model is deliberately not an MLP over the image. Images are reduced to
    a fixed 7x7 patch-sensor array; there is no direct sensor-to-output edge.
    Hidden cells receive sparse sensor and delayed recurrent messages. Ten
    output cells receive only hidden-cell messages.

    Args:
        seed: Local graph initialization seed. It does not seed PyTorch's
            shared-rule parameter initialization.
        n_hidden: Number of recurrent hidden cells.
        input_degree: Number of distinct patch sensors connected to each
            hidden cell.
        recurrent_degree: Number of distinct delayed hidden-cell inputs per
            hidden cell (self edges are excluded).
        steps: Number of synchronous recurrent ticks.
        memory_dim: Per-cell memory width; memory values are per example.
        edge_hidden: Width of the shared edge-rule MLP.
        cell_hidden: Width of the shared cell-rule MLP.

    Structural edits replace the per-edge weight Parameter. Make them before
    constructing the candidate's optimizer. Each forward call initializes
    fresh per-example cell and fast-edge state, so samples cannot share
    within-run plasticity state.
    """

    n_classes = 10
    n_sensor_ports = 49

    def __init__(
        self,
        seed: int = 0,
        n_hidden: int = 32,
        input_degree: int = 16,
        recurrent_degree: int = 4,
        steps: int = 3,
        memory_dim: int = 4,
        edge_hidden: int = 8,
        cell_hidden: int = 16,
    ) -> None:
        super().__init__()
        if n_hidden < max(5, recurrent_degree + 1):
            raise ValueError("n_hidden must exceed recurrent_degree and be at least 5")
        if not 1 <= input_degree <= self.n_sensor_ports:
            raise ValueError("input_degree must be in [1, 49]")
        if not 0 <= recurrent_degree < n_hidden:
            raise ValueError("recurrent_degree must be in [0, n_hidden)")
        if steps < 2:
            raise ValueError("steps must be at least 2 for sensor-to-readout propagation")
        if memory_dim < 1 or edge_hidden < 1 or cell_hidden < 1:
            raise ValueError("memory and rule widths must be positive")

        self.seed = int(seed)
        self.n_hidden = int(n_hidden)
        self.input_degree = int(input_degree)
        self.recurrent_degree = int(recurrent_degree)
        self.steps = int(steps)
        self.memory_dim = int(memory_dim)
        self.edge_hidden = int(edge_hidden)
        self.cell_hidden = int(cell_hidden)

        # Edge input: sender signal, receiver activity, receiver memory,
        # independent slow weight, independent fast weight, edge trace,
        # and a small learned edge-kind embedding.
        edge_feature_dim = 1 + 1 + self.memory_dim + 1 + 1 + 3
        self.edge_rule = nn.Sequential(
            nn.Linear(edge_feature_dim, self.edge_hidden),
            nn.Tanh(),
            nn.Linear(self.edge_hidden, 2),  # message correction, fast-weight delta
        )
        # The shared cell rule sees aggregate incoming messages and only the
        # receiving cell's own activity and memory.
        self.cell_rule = nn.Sequential(
            nn.Linear(2 + self.memory_dim, self.cell_hidden),
            nn.Tanh(),
            nn.Linear(self.cell_hidden, 1 + self.memory_dim),
        )
        self.edge_kind_embedding = nn.Embedding(3, 3)
        nn.init.normal_(self.edge_kind_embedding.weight, mean=0.0, std=0.05)
        self.logit_scale = nn.Parameter(torch.tensor(3.0))

        edges, weights = self._make_initial_graph(
            seed=self.seed,
            n_hidden=self.n_hidden,
            input_degree=self.input_degree,
            recurrent_degree=self.recurrent_degree,
        )
        self._install_graph(edges, weights)
        nn.init.zeros_(self.edge_rule[-1].bias)

    @staticmethod
    def _make_initial_graph(
        seed: int, n_hidden: int, input_degree: int, recurrent_degree: int
    ) -> tuple[list[tuple[int, int, int]], list[float]]:
        rng = random.Random(seed)
        edges: list[tuple[int, int, int]] = []
        for dst in range(n_hidden):
            for src in rng.sample(range(LocalNeuronClassifier.n_sensor_ports), input_degree):
                edges.append((EDGE_INPUT, src, dst))
        for dst in range(n_hidden):
            candidates = [src for src in range(n_hidden) if src != dst]
            for src in rng.sample(candidates, recurrent_degree):
                edges.append((EDGE_RECURRENT, src, dst))
        for class_index in range(LocalNeuronClassifier.n_classes):
            dst = n_hidden + class_index
            for src in range(n_hidden):
                edges.append((EDGE_READOUT, src, dst))
        weights = [rng.uniform(-0.1, 0.1) for _ in edges]
        return edges, weights

    @property
    def n_cells(self) -> int:
        return self.n_hidden + self.n_classes

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.numel())

    def _install_graph(
        self, edges: Iterable[tuple[int, int, int]], weights: Iterable[float]
    ) -> None:
        edge_list = list(edges)
        weight_list = list(weights)
        if len(edge_list) != len(weight_list):
            raise ValueError("each edge must have exactly one independent weight")
        self._validate_edges(edge_list)

        previous_weight = getattr(self, "edge_weight", None)
        device = previous_weight.device if previous_weight is not None else torch.device("cpu")
        dtype = previous_weight.dtype if previous_weight is not None else torch.float32
        src = torch.tensor([edge[1] for edge in edge_list], dtype=torch.long, device=device)
        dst = torch.tensor([edge[2] for edge in edge_list], dtype=torch.long, device=device)
        kind = torch.tensor([edge[0] for edge in edge_list], dtype=torch.long, device=device)
        if "edge_src" in self._buffers:
            self.edge_src = src
            self.edge_dst = dst
            self.edge_kind = kind
        else:
            self.register_buffer("edge_src", src)
            self.register_buffer("edge_dst", dst)
            self.register_buffer("edge_kind", kind)

        degree = torch.bincount(dst, minlength=self.n_cells).to(dtype=dtype)
        degree_scale = degree.clamp_min(1).rsqrt()
        if "edge_degree_scale" in self._buffers:
            self.edge_degree_scale = degree_scale
        else:
            self.register_buffer("edge_degree_scale", degree_scale)
        self.edge_weight = nn.Parameter(
            torch.tensor(weight_list, dtype=dtype, device=device)
        )

    def _validate_edges(self, edges: Iterable[tuple[int, int, int]]) -> None:
        for kind, src, dst in edges:
            if kind == EDGE_INPUT:
                if not 0 <= src < self.n_sensor_ports or not 0 <= dst < self.n_hidden:
                    raise ValueError("input edges must connect patch sensors to hidden cells")
            elif kind == EDGE_RECURRENT:
                if not 0 <= src < self.n_hidden or not 0 <= dst < self.n_hidden:
                    raise ValueError("recurrent edges must connect hidden cells")
                if src == dst:
                    raise ValueError("self recurrent edges are not allowed")
            elif kind == EDGE_READOUT:
                if not 0 <= src < self.n_hidden or not self.n_hidden <= dst < self.n_cells:
                    raise ValueError("readout edges must connect hidden cells to output cells")
            else:
                raise ValueError(f"unknown edge kind: {kind}")

    def _edges_and_weights(self) -> tuple[list[tuple[int, int, int]], list[float]]:
        edges = list(
            zip(
                self.edge_kind.detach().cpu().tolist(),
                self.edge_src.detach().cpu().tolist(),
                self.edge_dst.detach().cpu().tolist(),
            )
        )
        weights = self.edge_weight.detach().cpu().tolist()
        return edges, weights

    def _sensor_values(self, images: Tensor) -> Tensor:
        if images.ndim == 2 and images.shape[1] == 28 * 28:
            images = images.reshape(-1, 1, 28, 28)
        elif images.ndim == 3 and tuple(images.shape[-2:]) == (28, 28):
            images = images.unsqueeze(1)
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("images must have shape [B,1,28,28], [B,28,28], or [B,784]")
        # Fixed 4x4 average pooling creates 49 non-learned patch sensor ports.
        return F.avg_pool2d(images, kernel_size=4, stride=4).flatten(1)

    def _process_edges(
        self,
        sensors: Tensor,
        activity: Tensor,
        memory: Tensor,
        fast_weight: Tensor,
        edge_trace: Tensor,
        plastic: bool,
    ) -> tuple[Tensor, Tensor]:
        """Process every incoming edge independently from sender/receiver/edge state."""
        is_input = self.edge_kind == EDGE_INPUT
        sensor_idx = self.edge_src.clamp(0, self.n_sensor_ports - 1)
        cell_idx = self.edge_src.clamp(0, self.n_hidden - 1)
        sensor_signal = sensors[:, sensor_idx]
        cell_signal = activity[:, cell_idx]
        sender_signal = torch.where(is_input.unsqueeze(0), sensor_signal, cell_signal)

        receiver_activity = activity[:, self.edge_dst]
        receiver_memory = memory[:, self.edge_dst]
        if plastic:
            fast_used = fast_weight
            trace_used = edge_trace
        else:
            fast_used = torch.zeros_like(fast_weight)
            trace_used = torch.zeros_like(edge_trace)
        effective_weight = self.edge_weight.unsqueeze(0) + fast_used
        kind_features = self.edge_kind_embedding(self.edge_kind).unsqueeze(0)
        kind_features = kind_features.expand(sensors.shape[0], -1, -1)
        features = torch.cat(
            (
                sender_signal.unsqueeze(-1),
                receiver_activity.unsqueeze(-1),
                receiver_memory,
                effective_weight.unsqueeze(-1),
                trace_used.unsqueeze(-1),
                kind_features,
            ),
            dim=-1,
        )
        rule_output = self.edge_rule(features)
        # Each edge emits its own message. Same-source edges differ through
        # their own weight/history and the receiving cell's local state.
        messages = (
            effective_weight * sender_signal
            + 0.1 * torch.tanh(rule_output[..., 0])
        )
        if plastic:
            delta = torch.tanh(rule_output[..., 1])
            fast_next = torch.clamp(fast_weight + 0.01 * delta, min=-0.25, max=0.25)
            # A distinct correlation trace is retained for every edge.
            trace_next = 0.9 * edge_trace + 0.1 * sender_signal * receiver_activity
        else:
            fast_next = torch.zeros_like(fast_weight)
            trace_next = torch.zeros_like(edge_trace)
        return messages, fast_next, trace_next

    def _tick(
        self,
        sensors: Tensor,
        activity: Tensor,
        memory: Tensor,
        fast_weight: Tensor,
        edge_trace: Tensor,
        plastic: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one synchronous tick; all edges read pre-tick receiver/source state."""
        messages, fast_next, trace_next = self._process_edges(
            sensors, activity, memory, fast_weight, edge_trace, plastic
        )
        aggregate = messages.new_zeros((messages.shape[0], self.n_cells))
        aggregate = aggregate.index_add(1, self.edge_dst, messages)
        aggregate = aggregate * self.edge_degree_scale.unsqueeze(0)
        cell_input = torch.cat(
            (aggregate.unsqueeze(-1), activity.unsqueeze(-1), memory), dim=-1
        )
        cell_output = self.cell_rule(cell_input)
        next_activity = 0.5 * activity + 0.5 * torch.tanh(cell_output[..., 0])
        next_memory = 0.9 * memory + 0.1 * torch.tanh(cell_output[..., 1:])
        return next_activity, next_memory, fast_next, trace_next

    def forward(self, images: Tensor, plastic: bool = True) -> Tensor:
        if not isinstance(plastic, bool):
            raise TypeError("plastic must be a bool")
        sensors = self._sensor_values(images)
        batch_size = sensors.shape[0]
        activity = sensors.new_zeros((batch_size, self.n_cells))
        memory = sensors.new_zeros((batch_size, self.n_cells, self.memory_dim))
        fast_weight = sensors.new_zeros((batch_size, self.n_edges))
        edge_trace = sensors.new_zeros((batch_size, self.n_edges))
        for _ in range(self.steps):
            activity, memory, fast_weight, edge_trace = self._tick(
                sensors, activity, memory, fast_weight, edge_trace, plastic=plastic
            )
        logits = activity[:, self.n_hidden :] * self.logit_scale.clamp(1.0, 10.0)
        return logits
