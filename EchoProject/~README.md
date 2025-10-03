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
- **Derived dataset:** [PhysioNet Echo Note to Num v1.0.0](https://physionet.org/content/echo-note-to-num/1.0.0/), containing aligned notes and structured echocardiographic features.
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

**Note on prompt engineering:** The prompt-only approach used in this comparison could likely be improved with more sophisticated prompt engineering techniques (e.g., chain-of-thought prompting, few-shot examples, structured output constraints). However, even with such improvements, the fundamental challenge remains: prompt-only approaches require careful engineering for each new task and lack the consistency needed for production medical systems. The comparison demonstrates the baseline difficulty of the task and the value of fine-tuning, though the absolute prompt-only performance could potentially be enhanced with additional prompt optimization effort.

## Results

### Performance Comparison (Strict Evaluation)

**Strict evaluation methodology:** Failed predictions (unparseable outputs) are counted as incorrect, reflecting real-world deployment scenarios where any parsing failure would prevent clinical use.

#### Overall Performance Summary

| Method           | Average Per-Label Accuracy | Exact Match Accuracy | Failed Predictions |
|------------------|---------------------------|----------------------|-------------------|
| **Fine-tuned SFT**   | **99.88%** | **98.08%** | **0.02%** (19/125,552) |
| **Prompt-only**      | **9.87%** | **0.00%** | **51.23%** (64,319/125,552) |

#### Per-Label Accuracy Comparison

| Cardiac Feature | Fine-tuned Accuracy | Prompt-only Accuracy | Improvement |
|----------------|---------------------|---------------------|-------------|
| LA_cavity | 99.82% | 8.02% | +91.80% |
| RA_dilated | 99.98% | 9.47% | +90.51% |
| LV_systolic | 99.85% | 3.22% | +96.63% |
| LV_cavity | 99.94% | 6.99% | +92.95% |
| LV_wall | 99.83% | 12.70% | +87.13% |
| RV_cavity | 99.95% | 10.94% | +89.01% |
| RV_systolic | 99.89% | 12.65% | +87.24% |
| AV_stenosis | 99.86% | 4.54% | +95.32% |
| MV_stenosis | 99.91% | 9.44% | +90.47% |
| TV_regurgitation | 99.91% | 9.88% | +90.03% |
| TV_stenosis | 99.97% | 12.06% | +87.91% |
| TV_pulm_htn | 99.92% | 12.03% | +87.89% |
| AV_regurgitation | 99.56% | 13.26% | +86.30% |
| MV_regurgitation | 99.59% | 15.69% | +83.90% |
| RA_pressure | 99.98% | 9.93% | +90.06% |
| LV_diastolic | 99.85% | 8.76% | +91.09% |
| RV_volume_overload | 99.95% | 10.64% | +89.32% |
| RV_wall | 99.97% | 7.28% | +92.69% |
| RV_pressure_overload | 99.94% | 10.03% | +89.91% |
| **Average** | **99.88%** | **9.87%** | **+90.01%** |

### Key Findings

1. **Fine-tuning provides dramatic improvement:** The fine-tuned model improved accuracy by **~90 percentage points** across all labels, with improvements ranging from +83.90% to +96.63%.

2. **Reliability is critical:** Fine-tuning reduced failed predictions from 51.23% to 0.02%, ensuring consistent, parseable structured output suitable for clinical deployment.

3. **Robustness across features:** The fine-tuned model achieved >99.5% accuracy on all 19 labels, demonstrating consistent performance across diverse cardiac assessments. Even the most challenging labels (valve regurgitation features) exceeded 99.5% accuracy.

4. **Near-perfect extraction:** With 98.08% exact match accuracy, the fine-tuned model correctly extracted all 19 features simultaneously in the vast majority of cases, while the prompt-only model achieved 0% exact matches.

5. **Consistent superiority:** Fine-tuning improved performance on every single label without exception, with the smallest improvement still exceeding 83 percentage points.

### Critical Insight

For specialized medical text extraction tasks, **prompt engineering alone is insufficient for production deployment**. Despite detailed instructions and medical context, the prompt-only model:
- Failed to produce parseable output in more than half of all cases
- Achieved less than 16% accuracy on any individual label (best case: MV_regurgitation at 15.69%)
- Never correctly extracted all 19 features from a single report
- Showed high variability in performance across different cardiac features

While better prompt engineering might improve these baseline results, fine-tuning is essential for production-ready medical NLP systems requiring consistent, reliable structured data extraction. Even a relatively small 2B parameter model, when properly fine-tuned, dramatically outperforms zero-shot prompting approaches on domain-specific tasks.

## Technical Implementation

**Key technical decisions:**
- Used instruction-tuned base model (Gemma-2B-it) rather than base Gemma for better prompt following
- Formatted data as conversational turns matching the model's instruction format
- Applied causal language modeling (next-token prediction) rather than classification heads
- Truncated inputs at 2048 tokens to balance context length with training efficiency
- Used bfloat16 precision for memory efficiency without significant accuracy loss

## Repository Contents

- `echo_note_training_final.ipynb` — Complete fine-tuning workflow including data preparation, training configuration, and model training
- `Gemma_prompt_only_echo_label.ipynb` — Prompt-based experiments with regex post-processing
- `Echo_Fine_Tunning_vs_Prompt_Stats.ipynb` — Comprehensive performance comparison with strict evaluation methodology
- `Analyze_Echo_Performance.ipynb` — Model evaluation and visualization of results

## Future Work

- Optimize prompt engineering strategies (chain-of-thought, few-shot, structured output) to establish stronger prompt-only baseline
- Compare against domain-specific medical LLMs (e.g., Med-PaLM, ClinicalGPT)
- Evaluate larger model sizes (Gemma-9B, Gemma-27B) for potential accuracy improvements
- Explore parameter-efficient fine-tuning (LoRA, QLoRA) for resource-constrained environments
- Apply approach to more diverse, free-form clinical text beyond echocardiography
- Extend to other medical report types (radiology, pathology, discharge summaries)

## References

- **Dataset:** [Echo Note to Num v1.0.0](https://physionet.org/content/echo-note-to-num/1.0.0/) - PhysioNet
- **MIMIC Dataset Publication:** Alistair E.W. Johnson, Tom J. Pollard, Steven J. Berkowitz, Nathaniel R. Greenbaum, Matthew P. Lungren, Chih-ying Deng, Roger G. Mark, Steven Horng. "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports." *Scientific Data* 6, 317 (2019). DOI: 10.1038/s41597-019-0322-0
- **Base Model:** [Gemma-2B-it (Google)](https://ai.google.dev/gemma)



## Data Access

The Echo Note to Num dataset is available through PhysioNet and requires completion of a brief training course on human subjects research. Access the dataset at: https://physionet.org/content/echo-note-to-num/1.0.0/

---

## In summary 
This project demonstrates that fine-tuning is essential for extracting structured data from specialized medical text. Even a small, fine-tuned Gemma-2B model achieves near-perfect accuracy (99.88%) and reliability (0.02% failure rate), while the same model using only prompts achieves less than 10% accuracy with 51% failure rate. This work provides a practical blueprint for building production-ready medical NLP systems, showing that modest computational resources (single A100, ~2 hours training) can yield clinically meaningful results when combined with proper fine-tuning methodology.
