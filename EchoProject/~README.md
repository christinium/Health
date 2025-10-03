# Echo Project

Echocardiography reports are essential for understanding cardiac health, but their information is typically embedded in free-text notes that are difficult to process computationally. This project demonstrates how to transform these notes into structured data using machine learning and large language models (LLMs).

## Motivation

- **Clinical relevance:** Echocardiography findings are key indicators of cardiovascular risk and outcomes.
- **Technical challenge:** Clinical notes are lengthy, noisy, and heterogeneous. Extracting structured features requires handling unstructured text at scale.
- **Goal:** Compare the performance of fine-tuned LLMs vs. prompt-only approaches in extracting echocardiography features.

## Task Description

The task involves extracting **19 cardiac features** from unstructured echocardiogram reports and classifying each feature using a standardized coding schema. These features include:

- Left atrium (LA) cavity size, right atrium (RA) dilation
- Left ventricle (LV) systolic function, cavity size, wall thickness
- Right ventricle (RV) cavity, systolic function, wall thickness
- Valve conditions: aortic (AV), mitral (MV), and tricuspid (TV) stenosis and regurgitation
- Pulmonary hypertension, RA pressure
- LV diastolic function, RV volume/pressure overload

## Data

- **Source:** MIMIC-III echocardiography notes (restricted access).
- **Derived dataset:** PhysioNet Echo Note to Num, containing aligned notes and structured echocardiographic features.
- **Total samples:** 45,794 echocardiogram reports (44,047 after cleaning)
- **Dataset splits:** 
  - Training: 30,831 samples (70%)
  - Validation: 6,608 samples (15%)
  - Test: 6,608 samples (15%)

**Preprocessing:**
- Extracted echocardiography findings section to reduce sequence length
- Removed duplicates and reports with missing data
- Filtered reports by length (100-4096 characters)
- Tokenized text using Gemma's tokenizer (max length: 2048 tokens)
- Formatted as instruction-following prompts with structured output

## Models and Training

### 1. Fine-Tuned Model (Supervised Fine-Tuning - SFT)
- **Base Model:** Gemma-2B-it (2 billion parameter instruction-tuned model)
- **Frameworks:** HuggingFace Transformers, PyTorch
- **Task:** Multi-label classification of 19 echocardiography findings from free text

**Training Configuration:**
- Optimizer: AdamW
- Learning rate: 2e-4
- Weight decay: 0.01
- Warmup steps: 500
- Epochs: 2
- Batch size: 8 per device
- Precision: bfloat16
- Hardware: Single NVIDIA A100 GPU (Colab environment)
- Training time: ~2 hours

**Data Format:**
```
<start_of_turn>user
Analyze this echocardiogram report and provide assessment values for each cardiac feature...
Report:
[Echo report text]<end_of_turn>
<start_of_turn>model
LA_cavity: 0
RA_dilated: 0
...
<end_of_turn>
```

**Evaluation Metrics:**
- Per-label accuracy
- Exact match accuracy (all 19 labels correct)
- Failed prediction rate

### 2. Prompt-Only Model
- **Base Model:** Gemma-2B-it (zero-shot prompting)
- **Method:** Detailed prompts with medical context instructing the model to output structured labels

**Challenges:**
- Outputs inconsistent in both format and terminology
- Required regex post-processing for normalization
- High failure rate (~50%) producing unparseable outputs

## Results

### Performance Comparison (Strict Evaluation)

**Strict evaluation methodology:** Failed predictions (unparseable outputs) are counted as incorrect, reflecting real-world deployment scenarios where any parsing failure would prevent clinical use.

| Method           | Average Per-Label Accuracy | Exact Match Accuracy | Failed Predictions |
|------------------|---------------------------|----------------------|-------------------|
| **Fine-tuned SFT**   | **99.88%** | **98.08%** | **0.02%** |
| **Prompt-only**      | **9.87%** | **0.00%** | **51.23%** |

