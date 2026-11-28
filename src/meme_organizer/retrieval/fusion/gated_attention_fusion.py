"""Механизм внимания с гейтированием по запросу для объединения каналов поиска."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import torch
from torch import nn

if TYPE_CHECKING:
    import numpy as np


class GatedAttentionFusion(nn.Module):
    """Вычисляет веса внимания к каналам поиска на основе эмбеддинга запроса и маски модальностей."""

    def __init__(self, query_dim: int = 384, num_channels: int = 4, hidden_dim: int = 64) -> None:
        """Инициализирует двухслойный гейт внимания для отображения запроса в логиты каналов."""
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        channel_scores: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Возвращает объединённые оценки (n_queries, n_docs) с учётом маски доступности каналов."""
        logits = self.gate(query_emb).unsqueeze(1)
        masked_logits = logits.masked_fill(~channel_mask.unsqueeze(0), -1e9)
        weights = torch.softmax(masked_logits, dim=-1)
        return (weights * channel_scores).sum(dim=-1)

    @classmethod
    def train_model(
        cls,
        query_train: np.ndarray,
        scores_train: np.ndarray,
        gt_train: list[int],
        channel_mask: np.ndarray,
        epochs: int = 40,
        lr: float = 5e-3,
    ) -> GatedAttentionFusion:
        """Обучает модель с функцией потерь CrossEntropy по списку документов."""
        model = cls(query_dim=query_train.shape[1], num_channels=scores_train.shape[2])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        q_t = torch.from_numpy(query_train).float()
        s_t = torch.from_numpy(scores_train).float()
        m_t = torch.from_numpy(channel_mask).bool()
        y_t = torch.tensor(gt_train, dtype=torch.long)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            fused = model(q_t, s_t, m_t) * 10.0
            loss = loss_fn(fused, y_t)
            loss.backward()
            opt.step()

        model.eval()
        return model

    def export_onnx(
        self,
        output_path: str,
        query_dim: int = 384,
        num_channels: int = 4,
    ) -> None:
        """Экспортирует модель в формат ONNX для мобильного инференса."""
        self.eval()
        q_dummy = torch.randn(1, query_dim)
        s_dummy = torch.rand(1, 10, num_channels)
        m_dummy = torch.ones(10, num_channels, dtype=torch.bool)
        torch.onnx.export(
            self,
            (q_dummy, s_dummy, m_dummy),
            output_path,
            input_names=["query_emb", "channel_scores", "channel_mask"],
            output_names=["fused_scores"],
            dynamic_axes={
                "query_emb": {0: "n_queries"},
                "channel_scores": {0: "n_queries", 1: "n_docs"},
                "channel_mask": {0: "n_docs"},
                "fused_scores": {0: "n_queries", 1: "n_docs"},
            },
            opset_version=17,
        )
        model_proto = onnx.load(output_path, load_external_data=True)
        if model_proto.ir_version > 9:
            model_proto.ir_version = 9
        onnx.save_model(model_proto, output_path, save_as_external_data=False)
        data_file = Path(f"{output_path}.data")
        if data_file.exists():
            data_file.unlink()
