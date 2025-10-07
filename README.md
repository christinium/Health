# Health Project Repository

This repository contains two distinct research projects exploring Large Language Model (LLM) capabilities for medical documentation: structured data extraction from imaging reports and detection of clinical omissions in patient notes.

## Projects

### 1. EchoProject

#### Overview
A fine-tuned LLM system that extracts 19 structured cardiac features from unstructured echocardiogram reports. The project demonstrates that task-specific fine-tuning dramatically outperforms prompt engineering for specialized medical text extraction, achieving near-perfect accuracy (99.88%) with minimal computational resources.

#### Technical Approach
- **Base Model**: Gemma-2B-it (2 billion parameter instruction-tuned model)
- **Methods Compared**: 
  - Zero-shot prompting (baseline)
  - Few-shot prompting with examples
  - Supervised fine-tuning on 30,831 labeled reports
- **Architecture**: Causal language modeling with instruction-following format
- **Training**: 2 epochs, bfloat16 precision, single NVIDIA A100 GPU (~2 hours)
- **Output**: Structured severity labels for 19 cardiac features including chamber sizes, valve conditions, and functional assessments

#### Key Innovation
Establishes empirical evidence that **fine-tuning is essential for production medical NLP**, even when sophisticated prompt engineering is applied. While few-shot prompting improved accuracy 3x over zero-shot (10% → 41%), fine-tuning achieved 2.4x better performance than few-shot (41% → 99.88%), demonstrating that domain-specific training cannot be replaced by prompt optimization alone.

#### Performance Metrics - Three-Way Comparison

| Method | Avg Per-Label Accuracy | Exact Match Accuracy | Failed Predictions |
|--------|----------------------|---------------------|-------------------|
| **Fine-tuned SFT** | **99.88%** | **98.08%** | **0.02%** |
| **Few-shot Prompting** | **40.84%** | **0.08%** | **~0%** |
| **Zero-shot Prompting** | **9.87%** | **0.00%** | **51.23%** |

**Key Findings:**
- **Prompt engineering helps**: Adding examples to prompts improved accuracy 3.1x (from 9.87% to 40.84%)
- **But fine-tuning is essential**: Fine-tuned model achieved 99.88% accuracy—59 percentage points above few-shot prompting
- **Reliability gap**: Only fine-tuning achieved production-ready exact match rate (98% vs 0.08%)
- **Consistency**: Fine-tuning delivered >99.5% accuracy across ALL 19 features; prompt-based approaches showed high variance (ranging from +6% to +48% improvement with examples)

#### Clinical Relevance
Automates extraction of structured data from free-text echo reports, enabling:
- Clinical decision support systems
- Research cohort identification and risk stratification
- Quality metrics extraction and population health analytics
- Automated database population for registry studies

#### Current Limitations
- **Dataset-specific**: Trained on MIMIC-III echocardiograms; generalization to other institutions unknown
- **Report complexity**: MIMIC-III reports have relatively standardized language; performance on more complex or varied documentation styles requires validation
- **Model size constraint**: Limited to 2B parameters due to computational resources; larger models may improve edge case handling

#### Dataset
**Echo Note to Num v1.0.0** from PhysioNet: 44,047 MIMIC-III echocardiogram reports with 19 labeled cardiac features across severity scales (0=Normal, 1=Mild, 2=Moderate, 3=Severe, -1=Not mentioned). Dataset access requires completion of CITI training on human subjects research.

#### Next Steps
- Validation on external datasets with more complex clinical language
- Evaluation of larger model variants (Gemma-7B, Gemma-9B) for potential accuracy gains
- Parameter-efficient fine-tuning (LoRA/QLoRA) for resource-constrained deployment
- Extension to other cardiology report types (stress tests, cardiac catheterization, CT angiography)
- Real-time clinical deployment and workflow integration studies

---

### 2. Omission Detection System

#### Overview
A zero-shot LLM system that detects clinically relevant omissions in medical documentation using a clinical safety-first evaluation framework. Unlike traditional NLP metrics (BLEU/ROUGE) that focus on textual similarity, this system evaluates completeness based on clinical impact, achieving sub-human error rates.

#### Goal & Objective
**Goal**: Improve the quality and completeness of medical documentation  
**Objective**: Automatically detect clinically relevant details from transcripts that are missing from the HPI section of medical notes

#### Key Innovation
Traditional metrics like BLEU and ROUGE are insufficient for clinical workflows because they measure surface-level similarity rather than semantic understanding and clinical relevance. This system prioritizes **clinical impact over linguistic similarity**, ensuring patient safety through a three-tiered risk classification framework.

#### Technical Approach
- **Model**: GPT-4 with zero-shot learning (no fine-tuning required)
- **Method**: Chain-of-thought prompting for enhanced reasoning
- **Validation**: Clinician-in-the-loop for accuracy verification
- **Pipeline**: 
  1. Extract omitted content using GPT-4
  2. Classify omissions by clinical importance
  3. Score and format outputs into structured CSVs

#### Omission Definition
An **omission** is the absence of important information that should have been included in the patient's medical record. Missing information qualifies as an omission only if it is:
- Clinically relevant and necessary for understanding the patient's condition
- Potentially impactful on clinical decision-making and patient care
- Beneficial for enhancing the patient-physician relationship

**Note**: Missing information that doesn't meet these criteria is NOT considered an omission.

#### Three-Tiered Risk Classification

**Critical**
- Missing information that significantly impacts clinical decision-making or patient safety
- Could lead to diagnostic errors or inappropriate treatment decisions

**Moderate**
- Important context or supplemental information not critical to immediate decision-making
- Includes factors affecting physician-patient relationship (trust, communication, understanding)
- Example: Not documenting patient concerns or treatment preferences

**Optional**
- Useful but not necessary for clinical decision-making
- Minor nuances that don't significantly impact understanding of patient's condition

#### Evaluation Metrics

**Basic Metrics:**
- **Errors per Note**: Average number of omissions per note
- **Errors per Length of Transcript**: Omissions normalized by transcript length

**Weighted Metrics** (prioritizes severity):
- Critical Errors: 2 points
- Moderate Errors: 1 point  
- Optional Errors: 0 points

*Example*: 2 critical + 1 moderate + 0 optional = (2×2) + (1×1) + (0×0) = **5 weighted points**

- **Weighted Errors per Note**: Severity-weighted errors per note
- **Weighted Errors per Length**: Weighted errors per 10,000 words

#### Clinical Relevance
Ensures that:
- Critical medical information is never omitted
- Generated summaries are clinically safe and actionable
- Patient care quality is maintained or improved
- Administrative burden is reduced without compromising safety

#### Current Status
**Proof of Concept** - Demonstrates viability of clinically-informed metrics for medical documentation evaluation

#### Next Steps
- Improve prompts by adding additional examples to enhance accuracy
- Clinical validation with a panel of clinicians to verify real-world applicability

#### Reference
For more details on the evaluation framework: [A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation](https://www.nature.com/articles/s41746-025-01670-7) - *npj Digital Medicine* (2025)
