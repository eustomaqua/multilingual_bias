# AdaMCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Multilingual Chain-of-Thought

This repository contains the **codebase and training data** for the paper:

**AdaMCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Multilingual Chain-of-Thought**

AdaMCoT proposes an adaptive multilingual Chain-of-Thought (CoT) framework that allows models to dynamically choose the most effective reasoning language and then produce final answers in the instruction language, enabling stronger cross-lingual factual reasoning and consistency.

---

## Repository Structure

```text
.
├── Scripts_For_Data_Creation/
│   ├── run_generate_cross_openhermes.py
│   ├── utils.py
│   ├── generate_instructions.py
│   ├── translate_instructions.py
│   └── ranking_instructions.py
│
├── Ablation_Study/
│   ├── Create_Bar_Chart_mTruth.py
│   ├── Logit_Lens_Analysis.py
│   └── UMAP_Analysis.py
│
├── Training_Data/
│   └── Download_Dataset.py
│
└── README.md

```
---

## Overall Functionality

The scripts facilitate two major functionalities:

### 1) Dataset Generation for Fine-tuning

- Generating multilingual instruction sets
- Obtaining responses from LLMs (including potentially translated / cross-lingual responses)
- Using a judge LLM (e.g., GPT-4o via Azure OpenAI) to score / rank these responses
- Constructing SFT or preference datasets based on these evaluations  
  (the **Infinity-Firefly / AdaMCoT-style** dataset generation pipeline is a prime example)

---

## Scripts For Data Creation (`Scripts_For_Data_Creation/`)

### `run_generate_cross_openhermes.py`

**Purpose**: Primary script for orchestrating LLM evaluations on various benchmarks and for some data generation tasks.

**Key Functions**:

- `generate_with_vllm_multiprocessing()`  
  Core function for generating text from prompts using vLLM across multiple GPUs.
- `generate_crosslingual_openhermes()`  
  Prepares a cross-lingual dataset from OpenHermes translations (potentially for SFT).
- `generate_eval_crosslingual_openhermes()`  
  Evaluates models on a cross-lingual OpenHermes dataset, comparing different model versions.
- `generate_eval_reasoning_datasets()`  
  Generates responses for reasoning datasets in multiple languages.
- `generate_consistency_eval_crosslingual_openhermes()`  
  Evaluates model consistency across English and Chinese versions of OpenHermes prompts.
- `generate_cmmlu_eval_model_responses()`, `generate_crosslogiqa_eval_model_responses()`,  
  `generate_mmmlu_eval_model_responses()`, `generate_crossmmlu_eval_model_responses()`,  
  `generate_indommlu_eval_model_responses()`  
  Generate model responses for specific benchmarks.
- `extract_and_eval_..._ans()`  
  Extracts final multiple-choice answers (A/B/C/D) and computes accuracy using an LLM evaluator.
- `generate_cross_alpaca_eval_model_responses()`  
  Generates responses for the Cross-AlpacaEval benchmark.
- `generate_and_eval_aime_2024()`, `generate_and_eval_gpqa_diamond_mcq()`, `generate_and_eval_bfcl_v3()`  
  End-to-end generation and evaluation for specialized benchmarks.
- `generate_gpt_consistency_eval_crosslingual_openhermes()`, `generate_gpt_eval_crosslingual_openhermes()`,  
  `generate_gpt_eval_cmmlu()`  
  Prepares GPT-4o batch requests for evaluation via Azure OpenAI.

**Note**: This file is a monolith containing both generation and evaluation logic for many scenarios. The `if __name__ == "__main__":` block shows typical entry points.

---

### `utils.py`

**Purpose**: Utility functions used across scripts.

**Key Functions**:

- JSONL I/O: `write_jsonl()`, `read_jsonl()`
- Azure OpenAI Batch API wrappers:  
  `create_batch()`, `retrieve_batch()`, `retrieve_batch_response()`, `list_batch()`,  
  `output_gpt_request_jsons_for_instructions()`, `submit_batches()`, `download_batches()`
- `generate_with_vllm_multiprocessing()`  
  Efficient multi-GPU generation with vLLM (data parallelism).
- `async_request_questions()`  
  High-throughput async requests to LLM APIs (local vLLM server / Azure / Gemini).
- Post-processing helpers: `postprocess_outputs_row()`, `postprocess_inputs_row()`
- Ranking parser: `extract_gpt_ranking_score()`
- Output cleaners: `postprocess_thinking_answer()`, `postprocess_mcot_answer()`
- Consistency metric: `crosslingual_consistency()`
- Math answer extractor: `extract_boxed_answer()`

---

### `generate_instructions.py`

**Purpose**: Generating instruction datasets from multiple sources and preparing them for LLM answering / translation.

**Key Functions**:

- Instruction preparation:  
  `_generate_reasoning_datasets_requests()`, `_generate_crossopenhermes_datasets()`, `_generate_crossopenhermes_requests()`
- Answer generation: `_generate_answers_via_models()`
- `generate_answers_infinity_firefly()`  
  Generates answers for Infinity-Firefly cross-lingual instructions (AdaMCoT-style pipeline component).
- Preference pair construction: `_generate_preference_pairs_from_all_instructions()`
- Dataset sharding and processing: functions like `shard_..._datasets()` / `generate_..._datasets()`  
  (Infinity-Instruct, OpenR1-Math, KodCode, OpenThoughts, etc.)

---

### `translate_instructions.py`

**Purpose**: Translation of LLM responses / instructions into multiple languages.

**Key Function**:

- `translate_llama_responses()`  
  Translates answers into target languages using an LLM (e.g., Llama-3 8B Instruct).

---

### `ranking_instructions.py`

**Purpose**: Evaluate and rank LLM-generated answers using a judge model (GPT-4o), supporting filtered SFT and preference data creation.

**Key Functions**:

- Judge prompts: `QC_SYSTEM_PROMPT`, `COMPARE_QC_SYSTEM_PROMT`, `POST_TRANSLATE_SYSTEM_PROMPT`
- `generate_ranking_gpt_requests()`  
  Prepares GPT-4o requests to score answers.
- `generate_llama3_responses_ranking_requests_for_infinity_firefly()`  
  Prepares GPT-4o comparison requests across multiple candidate answers (including translated candidates).
- `analyse_ranking_infinity_firefly()`  
  Parses ranking results from Azure batch outputs and aggregates scores.
- `generate_infinity_firefly_full_cot_datasets()`  
  Constructs the final SFT dataset with AdaMCoT-style formatting:
  - If the best answer uses cross-lingual “thinking”, include thinking language trace + final answer in instruction language
  - Otherwise use direct generation format
- `generate_post_translate_requests_for_perference_learning()`  
  Generates post-translation requests to create better preference pairs.

---

## 2) Ablation Study (`Ablation_Study/`)

- `Create_Bar_Chart_mTruth.py`  
  Visualizes **path selection distribution** in the AdaMCoT framework.

- `Logit_Lens_Analysis.py`  
  Visualizes intermediate-layer reasoning distributions using **Logit Lens**:  
  https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

- `UMAP_Analysis.py`  
  Visualizes semantic space distributions using **UMAP**.

---

## Training Data (`Training_Data/`)

- `Download_Dataset.py`  
  Downloads the dataset created in this work.

Dataset is hosted on Hugging Face:  
https://huggingface.co/datasets/ZhengWH01/AdaMCoT

---

## Model Fine-tuning

We perform **LoRA fine-tuning** using **LLaMA Factory**:  
https://github.com/hiyouga/LlamaFactory

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zheng2025adamcotrethinkingcrosslingualfactual,
      title={AdaMCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Multilingual Chain-of-Thought}, 
      author={Weihua Zheng and Xin Huang and Zhengyuan Liu and Tarun Kumar Vangani and Bowei Zou and Xiyan Tao and Yuhao Wu and Ai Ti Aw and Nancy F. Chen and Roy Ka-Wei Lee},
      year={2025},
      eprint={2501.16154},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.16154}, 
}
