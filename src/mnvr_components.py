"""Shared layers and parameter construction for MNVR V2."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .SetTransformer import SAB


class GCN(nn.Module):
    def __init__(self, adjacency, dim, dropout=0.3):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32).clone()
        if adjacency.ndim != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("A drug graph must be square.")
        adjacency.fill_diagonal_(1)
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1)
        self.register_buffer("adjacency", adjacency)
        self.node_features = nn.Parameter(torch.empty(adjacency.size(0), dim))
        nn.init.xavier_uniform_(self.node_features)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self):
        hidden = self.dropout(F.relu(self.adjacency @ self.node_features))
        return self.adjacency @ self.projection(hidden)


class AdjAttenAgger(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, drugs, substructures, relevance, mask):
        scores = self.query(drugs) @ self.key(substructures).t() / math.sqrt(self.dim)
        has_substructure = mask.any(dim=1, keepdim=True)
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1) * has_substructure.to(scores.dtype)
        return weights @ (substructures * relevance.unsqueeze(-1))


class MNVRBase(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, ddi_mask_H, dim=64,
                 patient_dropout=0.3, graph_dropout=0.3, gnn_layers=4,
                 use_molecular=True, use_history=True, ddi_memory=False):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("dim must be a positive multiple of 2.")
        self.vocab_size = tuple(vocab_size)
        self.dim = dim
        self.use_molecular = use_molecular
        self.use_history = use_history
        self.ddi_memory = ddi_memory
        medications = vocab_size[2]
        if tuple(ehr_adj.shape) != (medications, medications):
            raise ValueError("EHR graph must use the current medication vocabulary.")
        if tuple(ddi_adj.shape) != (medications, medications):
            raise ValueError("DDI graph must use the current medication vocabulary.")
        ddi = torch.as_tensor(ddi_adj, dtype=torch.float32)
        ddi = ((ddi > 0) | (ddi.t() > 0)).float()
        ddi.fill_diagonal_(0)
        self.register_buffer("ddi_adjacency", ddi)
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], dim) for i in range(2)])
        self.dropout = nn.Dropout(patient_dropout)
        self.encoders = nn.ModuleList([nn.GRU(dim, 2 * dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(4 * dim, dim))
        self.ehr_gcn = GCN(ehr_adj, dim, graph_dropout)
        if ddi_memory:
            self.ddi_gcn = GCN(ddi, dim, graph_dropout)
            self.ddi_scale = nn.Parameter(torch.tensor(-2.0))
        self.output = nn.Sequential(
            nn.ReLU(), nn.Linear(3 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, medications)
        )
        if use_molecular:
            from gnn import GNNGraph
            mask = torch.as_tensor(ddi_mask_H) > 0
            if mask.ndim != 2 or mask.size(0) != medications:
                raise ValueError("Substructure mask rows must match medications.")
            self.register_buffer("substructure_mask", mask)
            gnn_options = dict(num_layer=gnn_layers, emb_dim=dim, graph_pooling="mean",
                               drop_ratio=graph_dropout, gnn_type="gin", virtual_node=False)
            self.global_encoder = GNNGraph(**gnn_options)
            self.substruct_encoder = GNNGraph(**gnn_options)
            self.sab = SAB(dim, dim, 2, use_ln=True)
            self.substruct_rela = nn.Linear(dim, mask.size(1))
            self.aggregator = AdjAttenAgger(dim)
            self.molecular_norm = nn.LayerNorm(dim)
            self.molecular_query = nn.Linear(dim, dim)
            self.molecular_scale = nn.Parameter(torch.tensor(0.1))

    def encode_static(self, molecule_graphs=None, average_projection=None, substructure_graphs=None):
        """Recompute each optimizer step; reuse only during no-grad evaluation."""
        memory = self.ehr_gcn()
        if self.ddi_memory:
            memory = memory - torch.sigmoid(self.ddi_scale) * self.ddi_gcn()
        result = {"memory": memory}
        if self.use_molecular:
            if molecule_graphs is None or average_projection is None or substructure_graphs is None:
                raise ValueError("Molecular graphs and projection are required.")
            result["global"] = average_projection @ self.global_encoder(molecule_graphs)
            result["substructure"] = self.sab(self.substruct_encoder(substructure_graphs).unsqueeze(0)).squeeze(0)
        return result

    def ddi_penalty(self, logits):
        """Smooth expected interacting-pair penalty, not the thresholded DDI rate."""
        probability = torch.sigmoid(logits)
        pairs = probability.t() @ probability
        denominator = max(self.vocab_size[2] * (self.vocab_size[2] - 1), 1)
        return (pairs * self.ddi_adjacency).sum() / denominator
