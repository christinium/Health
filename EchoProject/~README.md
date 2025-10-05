# Echo Project: Fine-Tuning vs Prompt Engineering for Medical Text Extraction

## TL;DR

Fine-tuning a small 2B parameter model achieves **99.88% accuracy** in extracting structured cardiac features from echocardiogram reports. Prompt engineering with examples improves zero-shot performance from 10% to 41%, but still falls **59 percentage points short** of fine-tuning. For production medical NLP systems requiring reliable structured data extraction, fine-tuning is essential.

---

## Overview

Echocardiography reports contain critical cardiac health information embedded in free-text clinical notes. This project demonstrates how to transform these unstructured reports into structured, computable data by comparing three approaches:

1. **Zero-shot prompting** - Base model with task instructions only
2. **Few-shot prompting** - Adding example inputs/outputs to guide the model  
3. **Supervised fine-tuning** - Training the model on domain-specific data

**Key Finding:** While prompt engineering provides meaningful improvements (3x better than zero-shot), fine-tuning remains the only approach suitable for production medical systems, achieving near-perfect accuracy with minimal computational resources.

---

## Motivation

### Clinical Relevance
- Echocardiography findings are key indicators of cardiovascular risk and patient outcomes
- Structured echo data enables population health analytics, clinical decision support, and research at scale
- Manual extraction is time-consuming, expensive, and doesn't scale to millions of reports

### Technical Challenge
- Clinical notes are lengthy, noisy, and highly variable in terminology and format
- Medical domain requires high accuracy - errors can impact patient care
- Need reliable, consistent extraction suitable for automated clinical workflows

### Research Question
**Can prompt engineering alone achieve production-ready performance, or is fine-tuning necessary for specialized medical text extraction?**

---

## Task Description

Extract **19 cardiac assessment features** from unstructured echocardiogram reports and classify each using a standardized coding schema:

**Cardiac Features:**
- **Chambers:** LA cavity size, RA dilation, LV systolic function, LV cavity size, LV wall thickness, RV cavity, RV systolic function, RV wall thickness
- **Valves:** Aortic valve (AV) stenosis/regurgitation, Mitral valve (MV) stenosis/regurgitation, Tricuspid valve (TV) stenosis/regurgitation
- **Hemodynamics:** Pulmonary hypertension (TV_pulm_htn), RA pressure, LV diastolic function
- **Other:** RV volume overload, RV pressure overload

**Classification Schema:** Each feature is coded on a severity scale:
- `0` = Normal/None
- `1` = Mild  
- `2` = Moderate
- `3` = Severe/Significant
- `-1` = Not mentioned/Unable to assess

---

## Dataset

**Source:** MIMIC-III echocardiography notes via [PhysioNet Echo Note to Num v1.0.0](https://physionet.org/content/echo-note-to-num/1.0.0/)

**Dataset Statistics:**
- **Total samples:** 45,794 echocardiogram reports
- **After cleaning:** 44,047 reports
- **Training set:** 30,831 samples (70%)
- **Validation set:** 6,608 samples (15%)
- **Test set:** 6,608 samples (15%)

**Preprocessing Pipeline:**
1. Extracted echocardiography findings section to reduce sequence length
2. Removed duplicate reports and entries with missing labels
3. Filtered by text length (100-4,096 characters)
4. Tokenized using Gemma's tokenizer (max length: 2,048 tokens)
5. Formatted as instruction-following prompts with structured output

**Example Input (truncated):**
```
Left Atrium: Mild LA enlargement. 
Left Ventricle: Normal LV cavity size with moderate regional systolic dysfunction.
Right Ventricle: Normal RV chamber size and free wall motion.
Aortic Valve: Mild AS. Trace AR.
Mitral Valve: Mild MR.
...
```

**Expected Output:**
```
LA_cavity: 1
RA_dilated: 0
LV_systolic: 2
LV_cavity: 0
...
```

---

## Methods

### 1. Zero-Shot Prompting (Baseline)
- **Model:** Gemma-2B-it (instruction-tuned, 2B parameters)
- **Approach:** Detailed task description with medical context and output format instructions
- **No training data** - relies entirely on the model's pre-trained knowledge

### 2. Few-Shot Prompting (Enhanced Baseline)
- **Model:** Gemma-2B-it 
- **Approach:** Added example echo reports with correct structured outputs to the prompt
- **Number of examples:** 3-5 representative cases showing input→output mapping
- **No model updates** - examples provided in context only

### 3. Supervised Fine-Tuning (SFT)
- **Base Model:** Gemma-2B-it
- **Training Strategy:** Causal language modeling on structured extraction task
- **Frameworks:** HuggingFace Transformers, PyTorch

**Training Configuration:**
- **Optimizer:** AdamW (lr=2e-4, weight_decay=0.01)
- **Warmup:** 500 steps
- **Epochs:** 2
- **Batch size:** 8 per device
- **Precision:** bfloat16
- **Hardware:** Single NVIDIA A100 GPU
- **Training time:** ~2 hours
- **Trainable parameters:** All parameters (full fine-tuning)

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

---

## Results

### Performance Comparison (Strict Evaluation)

**Evaluation Methodology:** Failed predictions (unparseable outputs) are counted as incorrect, reflecting real-world deployment requirements where parsing failures prevent clinical use.

#### Overall Performance Summary

| Method | Avg Per-Label Accuracy | Exact Match Accuracy | Failed Predictions |
|--------|----------------------|---------------------|-------------------|
| **Fine-tuned SFT** | **99.88%** | **98.08%** | **0.02%** (19/125,552) |
| **Prompt + Examples** | **40.84%** | **0.08%** | **~0%** |
| **Prompt Only** | **9.87%** | **0.00%** | **51.23%** (64,319/125,552) |

#### Per-Label Accuracy: Three-Way Comparison

| Cardiac Feature | Fine-tuned | With Examples | Zero-shot | Best Improvement* |
|----------------|-----------|--------------|-----------|------------------|
| LA_cavity | 99.82% | 14.39% | 8.02% | +6.37% |
| RA_dilated | 99.98% | 35.71% | 9.47% | +26.24% |
| LV_systolic | 99.85% | 22.05% | 3.22% | +18.83% |
| LV_cavity | 99.94% | 51.07% | 6.99% | +44.08% |
| LV_wall | 99.83% | 24.82% | 12.70% | +12.12% |
| RV_cavity | 99.95% | 25.76% | 10.94% | +14.82% |
| RV_systolic | 99.89% | 31.87% | 12.65% | +19.22% |
| AV_stenosis | 99.86% | 51.13% | 4.54% | +46.60% |
| MV_stenosis | 99.91% | 57.72% | 9.44% | +48.27% |
| TV_regurgitation | 99.91% | 56.42% | 9.88% | +46.53% |
| TV_stenosis | 99.97% | 56.93% | 12.06% | +44.87% |
| TV_pulm_htn | 99.92% | 29.19% | 12.03% | +17.16% |
| AV_regurgitation | 99.56% | 39.75% | 13.26% | +26.50% |
| MV_regurgitation | 99.59% | 27.32% | 15.69% | +11.62% |
| RA_pressure | 99.98% | 54.18% | 9.93% | +44.25% |
| LV_diastolic | 99.85% | 53.57% | 8.76% | +44.81% |
| RV_volume_overload | 99.95% | 47.84% | 10.64% | +37.20% |
| RV_wall | 99.97% | 48.50% | 7.28% | +41.22% |
| RV_pressure_overload | 99.94% | 47.70% | 10.03% | +37.67% |

\* *Improvement from adding examples (Few-shot vs Zero-shot)*

---

### Key Findings

#### 1. Prompt Engineering Shows Value, But Has Limits
- **Adding examples improved accuracy 3.1x** (from 9.87% to 40.84%)
- Example-driven prompting worked especially well for valve assessments (MV_stenosis: +48%, TV_stenosis: +45%)
- However, even with examples, accuracy remained **59 percentage points below fine-tuning**
- Improvement varied dramatically by feature (ranging from +6% to +48%)

#### 2. Fine-Tuning Provides Order-of-Magnitude Improvement
- Fine-tuned model achieved **99.88% average accuracy** across all 19 labels
- All individual labels exceeded **99.5% accuracy** - even the most challenging features
- Improvement over few-shot prompting: **+59 percentage points**
- Improvement over zero-shot prompting: **+90 percentage points**

#### 3. Reliability is Critical for Clinical Deployment
- **Zero-shot:** 51% of outputs were unparseable (complete failure)
- **Few-shot:** Dramatically reduced failures to near 0%, but accuracy remained inadequate
- **Fine-tuned:** Only 0.02% failures (19 out of 125,552 predictions)

#### 4. Exact Match Accuracy Reveals the Real Gap
- **Fine-tuned:** 98.08% of reports had all 19 features correctly extracted
- **Few-shot:** 0.08% exact match (only 5 perfect extractions out of 6,608 reports)
- **Zero-shot:** 0.00% exact match (zero perfect extractions)

For clinical applications requiring complete, accurate feature sets, only fine-tuning delivers production-ready performance.

#### 5. The Consistency Problem
Prompt-only approaches show high variance across features:
- Some labels benefit greatly from examples (valve stenosis: +45-48%)
- Others show minimal improvement (LA_cavity: +6%, MV_regurgitation: +12%)
- Fine-tuning achieves >99.5% on ALL labels uniformly

---

### Performance Visualization

**The Performance Spectrum:**
```
Zero-shot:     ████                      9.87%  (Baseline)
                     ↓ +31pp with examples
Few-shot:      ████████████              40.84% (3.1x improvement)
                             ↓ +59pp with fine-tuning  
Fine-tuned:    ███████████████████████   99.88% (Production-ready)
```

**Gap Analysis:**
- **Prompt engineering helps:** Going from zero-shot to few-shot closed 31% of the gap to perfection
- **But fine-tuning is essential:** The remaining 59% gap requires learning from domain data
- **Total improvement:** Fine-tuning provides 10x better accuracy than zero-shot (90pp improvement)

---

## Repository Contents

- **`echo_note_training_final.ipynb`** — Complete fine-tuning workflow: data preparation, training configuration, model training
- **`Gemma_prompt_only_echo_label.ipynb`** — Zero-shot prompting experiments with regex post-processing  
- **`Gemma_few_shot_echo_label.ipynb`** — Few-shot prompting with examples
- **`Echo_Fine_Tuning_vs_Prompt_Stats.ipynb`** — Three-way performance comparison with strict evaluation
- **`Analyze_Echo_Performance.ipynb`** — Model evaluation, error analysis, and result visualization

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers
- Access to PhysioNet Echo Note to Num dataset (requires CITI training)

### Reproducing Results
1. Obtain dataset access from PhysioNet
2. Run preprocessing notebook to prepare training data
3. Execute training notebook (requires A100 GPU or equivalent)
4. Evaluate on test set using comparison notebook

---

## Future Directions

### Near-term Improvements
- **Optimize few-shot prompting:** Experiment with chain-of-thought reasoning, structured output schemas, and systematic example selection
- **Parameter-efficient fine-tuning:** Explore LoRA/QLoRA for resource-constrained environments
- **Larger models:** Evaluate Gemma-7B and Gemma-9B for potential accuracy gains

### Extended Applications  
- **Other medical reports:** Apply methodology to radiology reports, pathology notes, discharge summaries
- **Multi-modal integration:** Combine text extraction with DICOM image analysis
- **Real-time deployment:** Optimize inference for clinical workflow integration
- **Comparative studies:** Benchmark against domain-specific medical LLMs (Med-PaLM, ClinicalGPT)

### Research Questions
- At what dataset size does fine-tuning surpass few-shot prompting for medical NLP?
- Can hybrid approaches (prompt engineering + lightweight fine-tuning) achieve optimal efficiency?
- How does performance generalize to other medical specialties and report types?

---

## Data Access & Licensing

**Dataset:** The Echo Note to Num dataset is available through PhysioNet (requires completion of CITI human subjects research training).  
**Access:** https://physionet.org/content/echo-note-to-num/1.0.0/

**Base Model:** Gemma-2B-it is available under Gemma Terms of Use from Google.

---

## References

1. **Dataset Source:**  
   Johnson, A.E.W., et al. (2019). "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports." *Scientific Data* 6, 317. DOI: 10.1038/s41597-019-0322-0

2. **PhysioNet Echo Dataset:**  
   [Echo Note to Num v1.0.0](https://physionet.org/content/echo-note-to-num/1.0.0/)

3. **Base Model:**  
   [Gemma-2B-it (Google DeepMind)](https://ai.google.dev/gemma)
