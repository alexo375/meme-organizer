from pathlib import Path

import nbformat as nbf


def create_nb01():
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell("""# Meme Organizer: 01. Лексический и Dense Zero-Shot поиск (Бейзлайны)

**Цель исследования:**
1. Построение воспроизводимого baseline ранжирования на чистом датасете `data/gold/gold_dataset.jsonl` (318 мемов, 1312 пар запросов).
2. Сравнение лексического поиска (**BM25 / SQLite FTS5**) и плотных нейросетевых энкодеров (**`rubert-tiny2`**, **`multilingual-e5-small`**).
3. Профилирование On-Device характеристик (задержка инференса на CPU в мс, размер ONNX INT8 в МБ).
4. Качественный анализ ошибок (**Top-15 Worst Queries**) и исследование эффекта гибридного слияния (Late Fusion).
""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd

from meme_organizer.data_splits.splitter import DatasetSplitter
from meme_organizer.retrieval.pipeline import RetrievalPipeline
from meme_organizer.retrieval.baselines.bm25 import BM25RetrieverAdapter
from meme_organizer.retrieval.adapters.dense_text import DenseTextRetrieverAdapter
from meme_organizer.evaluation.metrics import RetrievalEvaluator

# Фиксация путей
GOLD_DATASET_PATH = Path("../data/gold/gold_dataset.jsonl")
SPLIT_PATH = Path("../data/splits/splits_v1.json")
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Проверка и загрузка данных без утечек (Leak-Free Split)"))

    nb.cells.append(
        nbf.v4.new_code_cell("""with open(SPLIT_PATH) as f:
    split_info = json.load(f)

print(f"Корпус мемов: {len(split_info['corpus_hashes'])} документов")
print(f"Train пары:   {len(split_info['train_pairs'])} запросов")
print(f"Val пары:     {len(split_info['val_pairs'])} запросов")
print(f"Test пары:    {len(split_info['test_pairs'])} запросов")
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Запуск Baseline 1: Лексический BM25 (аналог SQLite FTS5)"))

    nb.cells.append(
        nbf.v4.new_code_cell("""pipeline = RetrievalPipeline()
bm25_adapter = BM25RetrieverAdapter(model_name="bm25-baseline")

bm25_metrics, bm25_run_id = pipeline.run(
    adapter=bm25_adapter,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
    run_name="bm25-baseline_test",
)

pd.DataFrame([{
    "Model": "BM25 (Lexical)",
    "Recall@1": f"{bm25_metrics.recall_at_1:.4f}",
    "Recall@5": f"{bm25_metrics.recall_at_5:.4f}",
    "MRR@5": f"{bm25_metrics.mrr_at_5:.4f}",
    "NDCG@5": f"{bm25_metrics.ndcg_at_5:.4f}",
    "MLflow Run ID": bm25_run_id,
}])
""")
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Запуск Baseline 2: Dense Bi-Encoders (`rubert-tiny2` & `multilingual-e5-small`)"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""rubert_adapter = DenseTextRetrieverAdapter(
    model_name="rubert-tiny2",
    hf_model_id="cointegrated/rubert-tiny2",
    max_length=64,
    onnx_dir=Path("../models/onnx"),
)

rubert_metrics, rubert_run_id = pipeline.run(
    adapter=rubert_adapter,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
    run_name="rubert-tiny2_test",
)

e5_adapter = DenseTextRetrieverAdapter(
    model_name="multilingual-e5-small",
    hf_model_id="intfloat/multilingual-e5-small",
    max_length=64,
    query_prefix="query: ",
    passage_prefix="passage: ",
    onnx_dir=Path("../models/onnx"),
)

e5_metrics, e5_run_id = pipeline.run(
    adapter=e5_adapter,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
    run_name="multilingual-e5-small_test",
)
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Сравнительная сводная таблица из MLflow"))

    nb.cells.append(
        nbf.v4.new_code_cell("""client = mlflow.tracking.MlflowClient("http://localhost:5000")
exp = client.get_experiment_by_name("meme-organizer-retrieval")
runs = client.search_runs(exp.experiment_id, order_by=["attribute.start_time DESC"], max_results=10)

leaderboard_rows = []
for r in runs:
    m = r.data.metrics
    leaderboard_rows.append({
        "Run Name": r.data.tags.get("mlflow.runName", "unnamed"),
        "Recall@1": round(m.get("recall_at_1", 0.0), 4),
        "Recall@5": round(m.get("recall_at_5", 0.0), 4),
        "MRR@5": round(m.get("mrr_at_5", 0.0), 4),
        "Latency (ms)": round(m.get("cpu_latency_ms", 0.0), 2),
        "Size (MB)": round(m.get("model_size_mb", 0.0), 2),
        "Eval (s)": round(m.get("eval_inference_time_sec", 0.0), 3),
        "Run ID": r.info.run_id[:8],
    })

pd.DataFrame(leaderboard_rows)
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Анализ Top-15 Worst Queries (Качественный анализ ошибок)"))

    nb.cells.append(
        nbf.v4.new_code_cell("""bm25_csv = client.download_artifacts(bm25_run_id, "error_analysis/worst_queries.csv")
df_worst_bm25 = pd.read_csv(bm25_csv)
print("=== Топ худших запросов для BM25 ===")
display(df_worst_bm25[["query", "target_rank"]].head(8))

e5_csv = client.download_artifacts(e5_run_id, "error_analysis/worst_queries.csv")
df_worst_e5 = pd.read_csv(e5_csv)
print("=== Топ худших запросов для E5-small ===")
display(df_worst_e5[["query", "target_rank"]].head(8))
""")
    )

    out_path = Path("notebooks/01_baselines_and_metrics.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("01_baselines_and_metrics.ipynb создан успешно!")


def create_nb02():
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(r"""# Meme Organizer: 02. Contrastive Domain Fine-Tuning (`rubert-tiny2` & `multilingual-e5-small`)

**Постановка задачи:**
Плотные модели общего назначения (RuBERT, E5-small) уступают лексическому поиску на специфичном интернет-сленге мемов (*«несет бред»* $\leftrightarrow$ *«в палату быстро»*, *«согласен с другом»* $\leftrightarrow$ *«мой меч с тобой»*).

В данном исследовании мы проводим **Domain Adaptation (дообучение)**:
1. Обучение на 1021 Train-парах через **Multiple Negatives Ranking Loss (MNRL)** с In-Batch Negatives.
2. Применение легковесных **LoRA** адаптеров (параметры замороженного backbone не изменяются).
3. Валидация по эпохам на 138 парах из `val_pairs` без заглядывания в тестовый сплит.
4. Экспорт обученных моделей в мобильный формат **ONNX INT8**.
5. Финальная оценка на Test-сплите (153 запроса) и сравнение на лидерборде MLflow.
""")
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from meme_organizer.retrieval.pipeline import RetrievalPipeline
from meme_organizer.retrieval.adapters.finetuned_dense import FineTunedDenseAdapter

GOLD_DATASET_PATH = Path("../data/gold/gold_dataset.jsonl")
SPLIT_PATH = Path("../data/splits/splits_v1.json")
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Дообучение ультралегкой модели `rubert-tiny2` (28 МБ ONNX)"))

    nb.cells.append(
        nbf.v4.new_code_cell("""pipeline = RetrievalPipeline()

rubert_ft_adapter = FineTunedDenseAdapter(
    model_name="rubert-tiny2-finetuned",
    base_model_id="cointegrated/rubert-tiny2",
    epochs=4,
    lr=2e-4,
    batch_size=32,
    max_length=64,
    lora_r=8,
    lora_alpha=16,
    onnx_dir=Path("../models/onnx"),
)

rubert_ft_metrics, rubert_ft_run_id = pipeline.run(
    adapter=rubert_ft_adapter,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
    train_mode=True,
    run_name="rubert-tiny2-finetuned_test",
)
""")
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Дообучение флагманской мультиязычной модели `multilingual-e5-small` (112 МБ ONNX)"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell("""e5_ft_adapter = FineTunedDenseAdapter(
    model_name="multilingual-e5-small-finetuned",
    base_model_id="intfloat/multilingual-e5-small",
    epochs=4,
    lr=1e-4,
    batch_size=16,
    max_length=64,
    query_prefix="query: ",
    passage_prefix="passage: ",
    lora_r=8,
    lora_alpha=16,
    onnx_dir=Path("../models/onnx"),
)

e5_ft_metrics, e5_ft_run_id = pipeline.run(
    adapter=e5_ft_adapter,
    gold_dataset_path=GOLD_DATASET_PATH,
    split_path=SPLIT_PATH,
    split_name="test",
    train_mode=True,
    run_name="multilingual-e5-small-finetuned_test",
)
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Графики обучения по эпохам из MLflow"))

    nb.cells.append(
        nbf.v4.new_code_cell("""client = mlflow.tracking.MlflowClient("http://localhost:5000")

# Получаем историю метрик для e5_ft
loss_hist = client.get_metric_history(e5_ft_run_id, "train_loss")
val_mrr_hist = client.get_metric_history(e5_ft_run_id, "val_mrr_at_5")

epochs = [m.step for m in loss_hist]
losses = [m.value for m in loss_hist]
val_mrrs = [m.value for m in val_mrr_hist]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, losses, marker="o", color="tab:red")
ax1.set_title("Training Loss (MNRL)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot(epochs, val_mrrs, marker="s", color="tab:blue")
ax2.set_title("Validation MRR@5")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MRR@5")
ax2.grid(True)

plt.tight_layout()
plt.show()
""")
    )

    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Итоговый лидерборд экспериментов Домена 1"))

    nb.cells.append(
        nbf.v4.new_code_cell("""exp = client.get_experiment_by_name("meme-organizer-retrieval")
runs = client.search_runs(exp.experiment_id, order_by=["attribute.start_time DESC"])

rows = []
for r in runs:
    m = r.data.metrics
    rows.append({
        "Model": r.data.tags.get("mlflow.runName", "unnamed"),
        "Recall@1": round(m.get("recall_at_1", 0.0), 4),
        "Recall@5": round(m.get("recall_at_5", 0.0), 4),
        "MRR@5": round(m.get("mrr_at_5", 0.0), 4),
        "Train Time (s)": round(m.get("training_time_sec", 0.0), 2),
        "Eval Latency (ms)": round(m.get("cpu_latency_ms", 0.0), 2),
        "Size (MB)": round(m.get("model_size_mb", 0.0), 2),
    })

pd.DataFrame(rows)
""")
    )

    out_path = Path("notebooks/02_contrastive_finetuning.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("02_contrastive_finetuning.ipynb создан успешно!")


if __name__ == "__main__":
    create_nb01()
    create_nb02()
