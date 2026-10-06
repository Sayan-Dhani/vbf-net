from __future__ import annotations

from typing import List, Tuple, Optional

# Legacy fallback used only when PyGVBFGNN is instantiated directly without
# num_targets. Normal training/inference should call PyGVBFGNN.from_config(),
# which derives the target count from the YAML config.
try:
    from pyg_vbf_dataset import NUM_TARGETS as DATASET_NUM_TARGETS
except Exception:
    DATASET_NUM_TARGETS = 4

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    MessagePassing,
    MultiAggregation,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter, softmax


def _activation(name: str) -> type[nn.Module]:
    table = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unknown activation '{name}'. Allowed: {sorted(table)}")
    return table[key]


def _norm_layer(name: str | None, dim: int) -> nn.Module:
    if name is None:
        return nn.Identity()

    key = str(name).lower()
    if key in {"none", "identity", ""}:
        return nn.Identity()
    if key == "layernorm":
        return nn.LayerNorm(dim)
    if key == "batchnorm":
        return nn.BatchNorm1d(dim)

    raise ValueError("norm must be one of: LayerNorm, BatchNorm, none")


def mlp(
    dims: List[int],
    act: str | type[nn.Module] = "GELU",
    dropout: float = 0.0,
    norm: str | None = "LayerNorm",
) -> nn.Sequential:
    layers: list[nn.Module] = []

    act_cls = _activation(act) if isinstance(act, str) else act

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))

        if i < len(dims) - 2:
            layers.append(_norm_layer(norm, dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class EdgeConvWithAttr(MessagePassing):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden: int,
        dropout: float = 0.0,
        aggregation: tuple[str, ...] = ("sum", "mean", "max"),
        activation: str = "GELU",
        norm: str | None = "LayerNorm",
    ):
        self.aggregation = tuple(aggregation)
        aggr = (
            MultiAggregation(list(self.aggregation))
            if len(self.aggregation) > 1
            else self.aggregation[0]
        )
        super().__init__(aggr=aggr)

        msg_dim = hidden
        agg_dim = len(self.aggregation) * msg_dim

        self.edge_mlp = mlp(
            [edge_dim + 2 * node_dim + global_dim, hidden, edge_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )
        self.edge_norm = _norm_layer(norm, edge_dim)

        self.gate_mlp = mlp(
            [2 * node_dim + edge_dim + global_dim, hidden, 1],
            act=activation,
            dropout=dropout,
            norm=None,
        )

        self.msg_mlp = mlp(
            [2 * node_dim + edge_dim + global_dim, hidden, msg_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )

        self.upd_mlp = mlp(
            [node_dim + agg_dim + global_dim, hidden, node_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )
        self.upd_norm = _norm_layer(norm, node_dim)
        self.res_proj = nn.Linear(node_dim, node_dim, bias=False)

        self.global_mlp = mlp(
            [global_dim + 2 * node_dim + 2 * edge_dim, hidden, global_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )
        self.global_norm = _norm_layer(norm, global_dim)
        self.global_res = nn.Linear(global_dim, global_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        ug: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]

        u_edge = ug[batch[src]]
        u_node = ug[batch]

        edge_in = torch.cat([edge_attr, x[src], x[dst], u_edge], dim=-1)
        edge_attr_new = self.edge_norm(self.edge_mlp(edge_in) + edge_attr)

        gate_in = torch.cat([x[dst], x[src], edge_attr_new, u_edge], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))

        agg = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr_new,
            u_edge=u_edge,
            gate=gate,
        )

        x_new = self.upd_mlp(torch.cat([x, agg, u_node], dim=-1))
        x_new = self.upd_norm(x_new + self.res_proj(x))

        node_mean = global_mean_pool(x_new, batch)
        node_max = global_max_pool(x_new, batch)

        edge_batch = batch[src]
        edge_mean = scatter(
            edge_attr_new,
            edge_batch,
            dim=0,
            dim_size=ug.size(0),
            reduce="mean",
        )
        edge_max = scatter(
            edge_attr_new,
            edge_batch,
            dim=0,
            dim_size=ug.size(0),
            reduce="max",
        )

        u_new = self.global_norm(
            self.global_res(ug)
            + self.global_mlp(
                torch.cat([ug, node_mean, node_max, edge_mean, edge_max], dim=-1)
            )
        )

        return x_new, edge_attr_new, u_new

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor,
        u_edge: Tensor,
        gate: Tensor,
    ) -> Tensor:
        m = self.msg_mlp(torch.cat([x_i, x_j, edge_attr, u_edge], dim=-1))
        return gate * m


class PyGVBFGNN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        node_dim: int = 128,
        edge_dim: int = 128,
        global_dim: int = 128,
        n_layers: int = 6,
        node_encoder_hidden: tuple[int, ...] = (128,),
        edge_encoder_hidden: tuple[int, ...] = (128,),
        global_encoder_hidden: tuple[int, ...] = (128,),
        head_hidden: tuple[int, ...] = (256, 128),
        mp_hidden: int | None = None,
        aggregation: tuple[str, ...] = ("sum", "mean", "max"),
        dropout: float = 0.0,
        pool: str = "mean+max",
        activation: str = "GELU",
        norm: str | None = "LayerNorm",
        num_targets: int | None = None,
        output_mode: str = "both",
        quantiles: tuple[float, ...] = (0.16, 0.50, 0.84),
        use_edge_pair_summary: bool = True,
        input_batchnorm: bool = False,
        input_batchnorm_affine: bool = True,
    ):
        super().__init__()

        if num_targets is None:
            # Backward-compatible legacy fallback. Prefer from_config() for any
            # YAML-driven target set, especially the 8-target p4 regression.
            num_targets = DATASET_NUM_TARGETS

        self.num_targets = int(num_targets)
        self.output_mode = str(output_mode)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.use_quantiles = self.output_mode in {"both", "quantile"}
        self.use_point = self.output_mode in {"both", "point"}
        self.n_quantiles = len(self.quantiles) if self.use_quantiles else 0
        self.n_heads = self.n_quantiles + (1 if self.use_point else 0)

        if self.n_heads <= 0:
            raise ValueError("At least one output head must be enabled.")

        # ── Optional input standardization (config-gated) ─────────────────────
        # A BatchNorm1d at the very input of each encoder z-scores the (already
        # log / signed-log1p-compressed) features using per-feature statistics
        # learned from training batches. Disabled by default: nn.Identity adds no
        # parameters or buffers, so the state_dict stays byte-identical to the
        # pre-feature-norm model and older checkpoints load with strict=True.
        self.input_batchnorm = bool(input_batchnorm)
        if self.input_batchnorm:
            self.node_input_norm = nn.BatchNorm1d(
                num_node_features, affine=input_batchnorm_affine
            )
            self.edge_input_norm = nn.BatchNorm1d(
                num_edge_features, affine=input_batchnorm_affine
            )
            self.global_input_norm = nn.BatchNorm1d(
                num_global_features, affine=input_batchnorm_affine
            )
        else:
            self.node_input_norm = nn.Identity()
            self.edge_input_norm = nn.Identity()
            self.global_input_norm = nn.Identity()

        self.node_encoder = mlp(
            [num_node_features] + list(node_encoder_hidden) + [node_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )
        self.edge_encoder = mlp(
            [num_edge_features] + list(edge_encoder_hidden) + [edge_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )
        self.global_encoder = mlp(
            [num_global_features] + list(global_encoder_hidden) + [global_dim],
            act=activation,
            dropout=dropout,
            norm=norm,
        )

        hidden_mp = (
            int(mp_hidden) if mp_hidden is not None else max(node_dim, edge_dim) * 2
        )

        self.conv_layers = nn.ModuleList(
            [
                EdgeConvWithAttr(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    global_dim=global_dim,
                    hidden=hidden_mp,
                    dropout=dropout,
                    aggregation=aggregation,
                    activation=activation,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )

        pool = pool.lower().strip()
        if pool not in {"mean", "max", "mean+max"}:
            raise ValueError(f"pool must be mean, max, or mean+max; got {pool}")
        self.pool = pool
        pool_out = 2 * node_dim if pool == "mean+max" else node_dim

        self.use_edge_pair_summary = bool(use_edge_pair_summary)
        if self.use_edge_pair_summary:
            pair_hidden = max(edge_dim, global_dim)
            self.edge_pair_score = mlp(
                [edge_dim + global_dim, pair_hidden, 1],
                act=activation,
                dropout=dropout,
                norm=norm,
            )
            self.edge_pair_proj = mlp(
                [edge_dim, edge_dim],
                act=activation,
                dropout=dropout,
                norm=norm,
            )
            self.pair_dim = edge_dim
        else:
            self.edge_pair_score = None
            self.edge_pair_proj = None
            self.pair_dim = 0

        reg_in = pool_out + global_dim + self.pair_dim
        reg_dims = [reg_in] + list(head_hidden) + [self.num_targets * self.n_heads]
        self.regressor = mlp(
            reg_dims,
            act=activation,
            dropout=dropout,
            norm=norm,
        )

    @classmethod
    def from_config(
        cls,
        *,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        cfg: dict,
    ) -> "PyGVBFGNN":
        m = cfg["model"]
        out = cfg["output"]

        return cls(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            node_dim=int(m.get("node_dim", 128)),
            edge_dim=int(m.get("edge_dim", 128)),
            global_dim=int(m.get("global_dim", 128)),
            n_layers=int(m.get("n_layers", 6)),
            node_encoder_hidden=tuple(m.get("node_encoder_hidden", [128])),
            edge_encoder_hidden=tuple(m.get("edge_encoder_hidden", [128])),
            global_encoder_hidden=tuple(m.get("global_encoder_hidden", [128])),
            head_hidden=tuple(m.get("head_hidden", [256, 128])),
            mp_hidden=m.get("mp_hidden", None),
            aggregation=tuple(m.get("aggregation", ["sum", "mean", "max"])),
            dropout=float(m.get("dropout", 0.0)),
            pool=str(m.get("pool", "mean+max")),
            activation=str(m.get("activation", "GELU")),
            norm=m.get("norm", "LayerNorm"),
            num_targets=len(cfg["targets"]),
            output_mode=str(out.get("mode", "both")),
            quantiles=tuple(float(q) for q in out.get("quantiles", [0.16, 0.50, 0.84])),
            use_edge_pair_summary=bool(m.get("use_edge_pair_summary", True)),
            input_batchnorm=bool(m.get("input_batchnorm", False)),
            input_batchnorm_affine=bool(m.get("input_batchnorm_affine", True)),
        )

    def forward(self, batch) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        u = batch.u
        b = batch.batch

        h = self.node_encoder(self.node_input_norm(x))
        e = self.edge_encoder(self.edge_input_norm(edge_attr))
        ug = self.global_encoder(self.global_input_norm(u))

        for conv in self.conv_layers:
            h, e, ug = conv(h, edge_index, e, ug, b)

        if self.pool == "mean":
            pooled = global_mean_pool(h, b)
        elif self.pool == "max":
            pooled = global_max_pool(h, b)
        else:
            pooled = torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], dim=-1)

        pieces = [pooled, ug]

        if self.use_edge_pair_summary:
            src = edge_index[0]
            edge_batch = b[src]
            u_edge_final = ug[edge_batch]

            score_in = torch.cat([e, u_edge_final], dim=-1)
            pair_logits = self.edge_pair_score(score_in).squeeze(-1)
            pair_alpha = softmax(pair_logits, edge_batch)

            pair_feat = self.edge_pair_proj(e)
            z_pair = scatter(
                pair_alpha.unsqueeze(-1) * pair_feat,
                edge_batch,
                dim=0,
                dim_size=ug.size(0),
                reduce="sum",
            )
            pieces.append(z_pair)

        combined = torch.cat(pieces, dim=-1)
        raw = self.regressor(combined)

        B = raw.shape[0]
        raw = raw.view(B, self.num_targets, self.n_heads)

        # For arbitrary quantile counts, enforce monotonic quantiles by sorting.
        # Head order remains: sorted quantiles, then point if enabled.
        out_parts = []

        if self.use_quantiles:
            q_raw = raw[:, :, : self.n_quantiles]
            q_sorted = torch.sort(q_raw, dim=-1).values
            out_parts.append(q_sorted)

        if self.use_point:
            point = raw[:, :, self.n_quantiles : self.n_quantiles + 1]
            out_parts.append(point)

        pred = torch.cat(out_parts, dim=-1)
        return pred.reshape(B, self.num_targets * self.n_heads)


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data

    from pyg_vbf_dataset import (
        NUM_EDGE_FEATURES,
        NUM_GLOBAL_FEATURES,
        NUM_NODE_FEATURES,
        NUM_TARGETS,
    )

    print(f"Node features  : {NUM_NODE_FEATURES}")
    print(f"Edge features  : {NUM_EDGE_FEATURES}")
    print(f"Global features: {NUM_GLOBAL_FEATURES}")
    print(f"Targets        : {NUM_TARGETS}")

    model = PyGVBFGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        num_global_features=NUM_GLOBAL_FEATURES,
        num_targets=NUM_TARGETS,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters     : {n_params:,}")

    B, P = 4, 5
    fake_data = []
    for _ in range(B):
        src = torch.arange(P).repeat_interleave(P - 1)
        dst = torch.cat(
            [
                torch.cat([torch.arange(P)[:i], torch.arange(P)[i + 1 :]])
                for i in range(P)
            ]
        )
        fake_data.append(
            Data(
                x=torch.randn(P, NUM_NODE_FEATURES),
                edge_index=torch.stack([src, dst], dim=0),
                edge_attr=torch.randn(P * (P - 1), NUM_EDGE_FEATURES),
                u=torch.randn(1, NUM_GLOBAL_FEATURES),
                y=torch.zeros(1, NUM_TARGETS),
            )
        )

    batch = Batch.from_data_list(fake_data)
    pred = model(batch)

    print(f"\nbatch.x    {tuple(batch.x.shape)}")
    print(f"batch.u    {tuple(batch.u.shape)}")
    print(f"batch.y    {tuple(batch.y.shape)}")
    print(f"pred       {tuple(pred.shape)}")

    assert pred.shape == (B, NUM_TARGETS * 4), f"Shape mismatch: {pred.shape}"
    print("\n✓ All shape checks passed.")
