# Echo Project

Echocardiography reports are essential for understanding cardiac health, but their information is typically embedded in free-text notes that are difficult to process computationally. This project demonstrates how to transform these notes into structured data using machine learning and large language models (LLMs).

## Motivation

- **Clinical relevance:** Echocardiography findings are key indicators of cardiovascular risk and outcomes.
- **Technical challenge:** Clinical notes are lengthy, noisy, and heterogeneous. Extracting structured features requires handling unstructured text at scale.
- **Goal:** Compare the performance of fine-tuned LLMs vs. prompt-only approaches in extracting echocardiography features.

## Data

- **Source:** MIMIC-III echocardiography notes (restricted access).
- **Derived dataset:** PhysioNet Echo Note to Num, containing aligned notes and structured echocardiographic features.

**Preprocessing:**
- Parsed reports and extracted the echocardiography findings section to reduce sequence length.
- Tokenized text using the model’s tokenizer.
- Split into train / tune / test sets.

## Models and Training

### 1. Fine-Tuned Model (Supervised Fine-Tuning - SFT)
- **Base Model:** Gemma (open LLM, lightweight transformer-based architecture).
- **Frameworks:** Hugging Face transformers, PyTorch.
- **Task:** Multi-label classification of echocardiography findings from free text.

**Training Details:**
- Optimizer: AdamW
- Scheduler: linear warmup + decay
- Epochs: 3
- Batch size: 8
- Learning rate: 5e-5
- Hardware: single GPU (Colab environment)

**Evaluation Metrics:**
- Accuracy
- Macro F1-score
- Per-class precision/recall

### 2. Prompt-Only Model
- **Base Model:** Gemma (zero-shot / few-shot prompting).
- **Method:** Designed prompts to instruct the model to output structured labels.

**Challenges:**
- Outputs inconsistent in both format and terminology.
- Required regex post-processing for normalization.
- Performance significantly lower than fine-tuned models.

## Results

| Method           | Accuracy      | Notes                                                      |
|------------------|--------------|------------------------------------------------------------|
| Fine-tuned SFT   | High (>90%)  | Stable performance across compartments, consistent output. |
| Prompt-only      | Lower (~60–70%) | Unreliable formatting, weaker per-class performance.     |

**Key finding:** Task-specific fine-tuning enables smaller LLMs to achieve strong performance, outperforming zero-shot prompting approaches.

## Repository Contents

- `echo_note_training_final.ipynb` — Fine-tuning workflow (data prep, training, validation)
- `Gemma_prompt_only_echo_label.ipynb` — Prompt-based experiments with regex post-processing
- `Analyze_Echo_Performance.ipynb` — Model evaluation and visualization of results

## Future Work

- Compare against domain-specific medical LLMs (e.g., Med-PaLM, ClinicalGPT)
- Explore handling of long-sequence notes with models like Longformer or LLaMA-2-Long
- Apply approach to more diverse, free-form clinical text beyond echocardiography

## References

- **Dataset:** PhysioNet Echo Note to Num
- **Publication:** [PubMed 40617839](https://pubmed.ncbi.nlm.nih.gov/40617839/)

---

**In summary:**  
This project highlights a practical workflow for fine-tuning open LLMs to extract structured data from clinical free-text, demonstrating that even modestly sized models can outperform prompt-only methods when applied to medical NLP tasks.