# Health Project Repository

This repository, Health, contains two distinct research projects that explore the capabilities of Large Language Models (LLMs) for the detection and mitigation of complex informational and behavioral issues within social contexts (physician visit), alongside the interpretation of imaging findings from diagnostic reports.

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

### 2. Omission Detection

#### Overview
An LLM-based tool that automatically identifies clinically relevant information missing from AI-generated medical notes by comparing History of Present Illness (HPI) sections against full patient visit transcripts. The system classifies omissions by severity (Critical, Moderate, Optional) and generates quantitative documentation quality scores.

#### Technical Approach
- **Model**: GPT-4-32k via Azure OpenAI
- **Architecture**: Two-stage prompt engineering strategy
  - Stage 1: Identifies missing clinical information from HPI by comparing against transcript
  - Stage 2: Classifies each omission by clinical severity with reasoning
- **Output**: Structured CSV with omission descriptions, severity levels, and clinical justifications

#### Key Innovation
Uses sequential LLM processing to separate omission detection from severity classification, reducing cognitive load on the model and improving classification accuracy. Employs role-based prompting with explicit clinical constraints to minimize hallucination and maintain relevance.

#### Clinical Relevance
- Evaluates completeness of AI-generated medical documentation
- Provides systematic, scalable quality assessment framework
- Generates explainable outputs with clinical reasoning
- Potential applications:
  - Auditing ambient AI scribes and documentation assistants
  - Training medical students on comprehensive history-taking
  - Quality improvement metrics for clinical documentation
  - Identifying systematic documentation gaps across providers

#### Current Limitations
- **Not clinically validated**: Requires expert physician review and inter-rater reliability testing
- **Prompt-based approach**: Unlike EchoProject, uses prompt engineering without fine-tuning; may benefit from supervised learning on validated omission labels
- **Non-deterministic**: Outputs vary across runs (temperature > 0)
- **No ground truth**: Severity classifications based on model reasoning, not validated clinical criteria
- **Context-dependent**: May not generalize across medical specialties or documentation styles

#### Next Steps
- Clinical validation study with multi-specialty expert physician review
- Establishment of gold-standard omission severity labels
- Assessment of clinical utility in real-world documentation workflows
- Exploration of fine-tuning approach (following EchoProject methodology) to improve consistency and accuracy
- Inter-rater reliability testing between LLM classifications and physician assessments



### 2. Omission Detection
Project Type: Proof of Concept | Status: Requires Clinical Validation

#### Overview
An LLM-based tool that automatically identifies clinically relevant information missing from AI-generated medical notes by comparing History of Present Illness (HPI) sections against full patient visit transcripts. The system classifies omissions by severity (Critical, Moderate, Optional) and generates quantitative documentation quality scores.

#### Technical Approach

- **Model**: GPT-4-32k via Azure OpenAI
- **Architecture**: Two-stage prompt engineering strategy
  - Stage 1: Identifies missing clinical information from HPI by comparing against transcript
  - Stage 2: Classifies each omission by clinical severity with reasoning
- **Output**: Structured CSV with omission descriptions, severity levels, and clinical justifications

#### Key Innovation
Uses sequential LLM processing to separate omission detection from severity classification, reducing cognitive load on the model and improving classification accuracy. Employs role-based prompting with explicit clinical constraints to minimize hallucination and maintain relevance.

#### Clinical Relevance
- Evaluates completeness of AI-generated medical documentation
- Provides systematic, scalable quality assessment framework
- Generates explainable outputs with clinical reasoning
- Potential applications:
  - Auditing ambient AI scribes
  - Training medical students
  - Quality improvement metrics

#### Current Limitations
- **Not clinically validated**: Requires expert physician review and inter-rater reliability testing
- **Non-deterministic**: Outputs vary across runs (temperature > 0)
- **No ground truth**: Severity classifications based on model reasoning, not validated criteria
- **Context-dependent**: May not generalize across specialties or documentation styles

#### Next Steps
Clinical validation study with multi-specialty expert review, establishment of gold-standard labels, and assessment of clinical utility in real-world documentation workflows.
