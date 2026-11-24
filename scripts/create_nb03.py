import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}

nb.cells.append(
    nbf.v4.new_markdown_cell(r"""# Meme Organizer: 03. Hybrid Retrieval & Cross-Encoder Re-ranking
В этом блокноте финализируется **текстовый домен поиска**:
1. **First-Stage Late Fusion**:
   - **RRF (Reciprocal Rank Fusion)**: бескомпромиссное объединение рангов BM25 и Fine-tuned E5 без параметров ($k=60$).
   - **Optuna Convex Score Combination**: калибровка веса $\alpha \in [0, 1]$ на валидационном сплите (`val`) с последующей честной оценкой на тесте (`test`).
2. **Second-Stage Re-ranking**:
   - Отбор кандидатов (топ-20) первой стадией и повторное ранжирование легковесным ONNX INT8 Cross-Encoder (`rubert-tiny-bilingual-nli`, размер 11.3 МБ).
   - Оценка выигрыша по качеству ($Recall@1$, $Recall@5$, $MRR@5$) и накладных расходов по задержке ($ms$).
""")
)

nb.cells.append(
    nbf.v4.new_code_cell(r"""import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd

from meme_organizer.data_splits.splitter import DatasetSplitter
from meme_organizer.retrieval.pipeline import RetrievalPipeline
from meme_organizer.retrieval.adapters.hybrid import HybridRetrieverAdapter
from meme_organizer.evaluation.metrics import RetrievalEvaluator

GOLD_DATASET_PATH = Path("../data/gold/gold_dataset.jsonl")
SPLIT_PATH = Path("../data/splits/splits_v1.json")
""")
)

nb.cells.append(nbf.v4.new_markdown_cell(r"""## 1. Проверка золотых сплитов"""))

nb.cells.append(
    nbf.v4.new_code_cell(r"""with open(SPLIT_PATH) as f:
    split_info = json.load(f)

print(f"Корпус мемов: {len(split_info['corpus_hashes'])} документов")
print(f"Train пары:   {len(split_info['train_pairs'])} запросов")
print(f"Val пары:     {len(split_info['val_pairs'])} запросов")
print(f"Test пары:    {len(split_info['test_pairs'])} запросов")
""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell(r"""## 2. Эксперимент 1: Late Fusion via Reciprocal Rank Fusion (RRF)
RRF стабилен к разнице шкал и распределений плотных и разреженных скоров:
$$RRF(d) = \sum_{m} \frac{1}{60 + rank_m(d)}$$
""")
)

nb.cells.append(
    nbf.v4.new_code_cell(r"""pipeline = RetrievalPipeline()
adapter_rrf = HybridRetrieverAdapter(mode="rrf", model_name="hybrid-rrf")

metrics_rrf, run_id_rrf = pipeline.run(
    adapter=adapter_rrf,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
)

pd.DataFrame([
    {
        "Model": "Hybrid RRF (BM25 + E5-FT)",
        "Recall@1": metrics_rrf.recall_at_1,
        "Recall@5": metrics_rrf.recall_at_5,
        "MRR@5": metrics_rrf.mrr_at_5,
        "NDCG@5": metrics_rrf.ndcg_at_5,
        "MLflow Run ID": run_id_rrf,
    }
])
""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell(r"""## 3. Эксперимент 2: Convex Score Combination с подбором веса $\alpha$ через Optuna
Подбор веса $\alpha$ происходит **строго на валидационном сплите** (`val_pairs`), после чего замороженный вес применяется к тесту (`test_pairs`).
""")
)

nb.cells.append(
    nbf.v4.new_code_cell(r"""adapter_optuna = HybridRetrieverAdapter(mode="optuna", model_name="hybrid-optuna")

metrics_optuna, run_id_optuna = pipeline.run(
    adapter=adapter_optuna,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
)

print(f"Подобранный вес alpha (Dense vs BM25): {adapter_optuna.alpha:.4f}")
pd.DataFrame([
    {
        "Model": f"Hybrid Convex (alpha={adapter_optuna.alpha:.2f})",
        "Recall@1": metrics_optuna.recall_at_1,
        "Recall@5": metrics_optuna.recall_at_5,
        "MRR@5": metrics_optuna.mrr_at_5,
        "NDCG@5": metrics_optuna.ndcg_at_5,
        "MLflow Run ID": run_id_optuna,
    }
])
""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell(r"""## 4. Эксперимент 3: Two-Stage Retrieval с Cross-Encoder Re-ranking
Отбираем топ-20 кандидатов первой стадией и реранжируем их с помощью `rubert-tiny-bilingual-nli` (INT8 ONNX).
""")
)

nb.cells.append(
    nbf.v4.new_code_cell(r"""adapter_ce = HybridRetrieverAdapter(mode="rerank_ce", model_name="hybrid-rerank-ce", rerank_top_k=20, alpha=adapter_optuna.alpha)

metrics_ce, run_id_ce = pipeline.run(
    adapter=adapter_ce,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
)

pd.DataFrame([
    {
        "Model": "Two-Stage (Hybrid + CE Reranker)",
        "Recall@1": metrics_ce.recall_at_1,
        "Recall@5": metrics_ce.recall_at_5,
        "MRR@5": metrics_ce.mrr_at_5,
        "NDCG@5": metrics_ce.ndcg_at_5,
        "MLflow Run ID": run_id_ce,
    }
])
""")
)

nb.cells.append(nbf.v4.new_markdown_cell(r"""## 5. Итоговый лидерборд текстового поиска (MLflow Comparison)"""))

nb.cells.append(
    nbf.v4.new_code_cell(r"""client = mlflow.tracking.MlflowClient(tracking_uri="http://localhost:5000")
exp = client.get_experiment_by_name("meme-organizer-retrieval")
runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time ASC"])

data = []
for r in runs:
    name = r.data.tags.get("mlflow.runName", "unnamed")
    m = r.data.metrics
    data.append({
        "Model": name,
        "Recall@1": m.get("recall_at_1", 0),
        "Recall@5": m.get("recall_at_5", 0),
        "MRR@5": m.get("mrr_at_5", 0),
        "NDCG@5": m.get("ndcg_at_5", 0),
        "P95 Latency (ms)": m.get("cpu_latency_p95_ms", 0),
        "Model Size (MB)": m.get("model_size_mb", 0),
        "Eval Time (s)": m.get("eval_inference_time_sec", 0),
    })

df_leaderboard = pd.DataFrame(data)
df_leaderboard.sort_values(by="MRR@5", ascending=False).reset_index(drop=True)
""")
)

with open("notebooks/03_hybrid_and_reranking.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Saved notebooks/03_hybrid_and_reranking.ipynb")
