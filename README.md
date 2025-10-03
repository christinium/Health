# Health Project Repository

This repository, Health, contains two distinct research projects that explore the capabilities of Large Language Models (LLMs) for the detection and mitigation of complex informational and behavioral issues within social contexts (physician visit), alongside the interpretation of imaging findings from diagnostic reports.

## Projects

### 1. EchoProject
Project Type: Research Implementation | Status: Technical Validation Complete
#### Overview
A fine-tuned LLM system that extracts 19 structured cardiac features from unstructured echocardiogram reports. The project demonstrates that task-specific fine-tuning dramatically outperforms zero-shot prompting for specialized medical text extraction, achieving near-perfect accuracy (99.88%) with minimal computational resources.

#### Technical Approach
- **Base Model**: Gemma-2B-it (2 billion parameter instruction-tuned model)
- **Method**: Supervised fine-tuning on 30,831 labeled echocardiogram reports
- **Architecture**: Causal language modeling with instruction-following format
- **Training**: 2 epochs, bfloat16 precision, single NVIDIA A100 GPU (~2 hours)
- **Output**: Structured labels for 19 cardiac features including chamber sizes, valve conditions, and functional assessments

#### Key Innovation
Demonstrates that small, domain-specific fine-tuned models can achieve production-ready performance for medical text extraction where prompt engineering fails. The fine-tuned 2B model improved accuracy by 90 percentage points over prompt-only approaches and reduced parsing failures from 51% to 0.02%, establishing fine-tuning as essential for reliable medical NLP systems.

#### Performance Metrics
- **Fine-tuned Model**: 99.88% per-label accuracy, 98.08% exact match (all 19 labels correct)
- **Prompt-only Baseline**: 9.87% per-label accuracy, 0% exact match, 51.23% parsing failures
- **Reliability**: <0.02% failed predictions, ensuring consistent structured output
- **Robustness**: >99.5% accuracy across all 19 cardiac features

#### Clinical Relevance
Automates extraction of structured data from free-text echo reports, enabling:
- Clinical decision support systems
- Research cohort identification
- Quality metrics extraction
- Automated database population

#### Current Limitations
- **Dataset-specific**: Trained on MIMIC-III echocardiograms; generalization to other institutions unknown
- **Structured reports only**: Designed for standardized echo report formatpercentages)
- **Simple language**: The reports have relatively simple language; generalization to more complicated language unknown

#### Dataset
Echo Note to Num v1.0.0 from PhysioNet (44,047 MIMIC-III echocardiogram reports with 19 labeled cardiac features). Dataset access requires completion of CITI training on human subjects research.

#### Next Steps
Validation on more complicated notes, evaluation of larger model variants (Gemma-9B) for potential accuracy improvements.


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