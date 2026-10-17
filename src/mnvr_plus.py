"""MNVR V2: code-level evidence and learned prescription continuation."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.mnvr_components import GCN, MNVRBase


class ResidualDrugGCN(GCN):
    """Retain drug identity when the co-prescription graph is dense."""
    def __init__(self, adjacency, dim, dropout):
        super().__init__(adjacency, dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self):
        return self.norm(self.node_features + super().forward())


class MNVRPlus(MNVRBase):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, ddi_mask_H, dim=64,
                 patient_dropout=0.3, graph_dropout=0.3, gnn_layers=4,
                 use_molecular=True, use_history=True, ddi_memory=False,
                 use_copy=True, use_direct=True):
        super().__init__(vocab_size, ehr_adj, ddi_adj, ddi_mask_H, dim,
                         patient_dropout, graph_dropout, gnn_layers,
                         use_molecular, use_history, ddi_memory)
        self.use_copy = use_copy
        self.use_direct = use_direct
        self.ehr_gcn = ResidualDrugGCN(ehr_adj, dim, graph_dropout)
        if ddi_memory:
            self.ddi_gcn = ResidualDrugGCN(ddi_adj, dim, graph_dropout)
        self.code_pooling = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))
            for _ in range(2)
        ])
        self.current_projection = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
        self.query_norm = nn.LayerNorm(dim)
        if use_direct:
            # Each diagnosis/procedure can contribute to each drug without having
            # to survive a mean pooling -> GRU -> graph-memory bottleneck.
            self.direct_heads = nn.ModuleList([nn.Embedding(vocab_size[i], vocab_size[2]) for i in range(2)])
            for head in self.direct_heads:
                nn.init.zeros_(head.weight)
        if use_history:
            self.recency_decay = nn.Parameter(torch.tensor(-1.0))
        if use_copy and use_history:
            self.copy_gate = nn.Linear(3 * dim, vocab_size[2])
            nn.init.normal_(self.copy_gate.weight, std=0.01)
            nn.init.constant_(self.copy_gate.bias, math.log(0.2 / 0.8))
        nn.init.normal_(self.output[-1].weight, std=0.01)

    @torch.no_grad()
    def initialize_prevalence(self, prevalence):
        """Called only for NEW runs, with frequencies computed from training patients."""
        prior = torch.as_tensor(prevalence, device=self.output[-1].bias.device, dtype=torch.float32)
        if prior.shape != self.output[-1].bias.shape:
            raise ValueError("Medication prior has the wrong shape.")
        prior = prior.clamp(1e-4, 1 - 1e-4)
        self.output[-1].bias.copy_(torch.logit(prior))

    def forward(self, visits, molecule_graphs=None, average_projection=None,
                substructure_graphs=None, static=None):
        if not visits:
            raise ValueError("At least one visit is required.")
        device = self.embeddings[0].weight.device
        sequences, pooled_sequences = [], []
        for field in range(2):
            pooled = []
            for visit in visits:
                codes = torch.as_tensor(visit[field], dtype=torch.long, device=device)
                if codes.numel():
                    embeddings = self.dropout(self.embeddings[field](codes))
                    attention = torch.softmax(self.code_pooling[field](embeddings), dim=0)
                    # Keep a mean path as well: attention need not rediscover all codes.
                    pooled.append(0.5 * (embeddings.mean(0) + (attention * embeddings).sum(0)))
                else:
                    pooled.append(self.embeddings[field].weight.new_zeros(self.dim))
            pooled = torch.stack(pooled)
            pooled_sequences.append(pooled)
            encoded, _ = self.encoders[field](pooled.unsqueeze(0))
            sequences.append(encoded.squeeze(0))
        current = self.current_projection(torch.cat(pooled_sequences, dim=-1))
        queries = self.query_norm(self.query(torch.cat(sequences, dim=-1)) + current)
        query = queries[-1:]
        if static is None:
            static = self.encode_static(molecule_graphs, average_projection, substructure_graphs)
        elif self.training:
            raise ValueError("Do not reuse cached graph features during training.")
        memory = static["memory"]
        fact1 = torch.softmax(query @ memory.t() / math.sqrt(self.dim), dim=-1) @ memory
        fact2, history_probability, history_context = fact1, None, None
        if self.use_history and len(visits) > 1:
            history_values = memory.new_zeros(len(visits) - 1, self.vocab_size[2])
            for index, visit in enumerate(visits[:-1]):
                history_values[index, visit[2]] = 1
            age = torch.arange(len(visits) - 2, -1, -1, device=device, dtype=memory.dtype)
            scores = query @ queries[:-1].t() / math.sqrt(self.dim)
            weights = torch.softmax(scores - F.softplus(self.recency_decay) * age, dim=-1)
            history_probability = weights @ history_values
            history_context = weights @ queries[:-1]
            fact2 = history_probability @ memory / history_probability.sum().clamp_min(1)
        logits = self.output(torch.cat([query, fact1, fact2], dim=-1))
        if self.use_direct:
            for field in range(2):
                codes = torch.as_tensor(visits[-1][field], dtype=torch.long, device=device)
                if codes.numel():
                    logits = logits + self.direct_heads[field](codes).sum(0, keepdim=True) / math.sqrt(codes.numel())
        if self.use_molecular:
            relevance = torch.sigmoid(self.substruct_rela(query)).squeeze(0)
            local = self.aggregator(static["global"], static["substructure"], relevance, self.substructure_mask)
            drugs = self.molecular_norm(static["global"] + local)
            logits = logits + self.molecular_scale * (self.molecular_query(query) @ drugs.t()) / math.sqrt(self.dim)
        if self.use_copy and self.use_history and history_probability is not None:
            gate_input = torch.cat([query, history_context, query - history_context], dim=-1)
            gate = torch.sigmoid(self.copy_gate(gate_input))
            probability = (1 - gate) * torch.sigmoid(logits) + gate * history_probability
            logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        return logits
