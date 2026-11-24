# Context for Antigravity / Gemini Agent

## Project Overview
This repository contains the Machine Learning search & retrieval pipeline for **Meme Organizer** (an on-device Android application for searching memes and video reactions).
Target Runtime: Android (ONNX Runtime Mobile, INT8 quantized, memory footprint < 150MB, latency < 50ms on CPU).
Development environment: Linux, Intel Core i7-12700H CPU (20 threads, no GPU).

## Architecture Blueprint & Critical Rules
Before implementing any experiments, models, or evaluation code, **you MUST read**:
[ROADMAP.md](file:///home/alex/Projects/meme-organizer/.agents/ROADMAP.md)

### Key Architectural Guidelines:
1. **Never build over-engineered trackers:**
   - Do NOT build bloated tracking builders or require 20 mandatory logging calls.
   - Use the **Template Method + Adapter** pattern (`RetrievalPipeline.run()` calls `adapter.fit()`, `adapter.encode_*()`, calculates metrics, profiles latency/size, and automatically logs to MLflow).
   - Only log: Config/Hyperparams, Metrics (MRR@5, Recall@1, Recall@5, NDCG@5), Performance (latency_ms, size_mb), and Error Analysis Table (Top-15 Worst Queries).
2. **Prevent Data Leakage:**
   - Always group related memes/scenes (`scene_group`) using `GroupShuffleSplit`.
   - Never split the corpus of memes; split query pairs into Train/Val/Test.
   - Fixed split must reside in `data/splits/splits_v1.json`.
3. **Data Source:**
   - Gold dataset: `data/gold/gold_dataset.jsonl` (318 records).
   - Media directory: `data/silver/`.

Refer to `.agents/ROADMAP.md` for the domain-by-domain experimental plan.
