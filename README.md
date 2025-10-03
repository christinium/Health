# Health Project Repository

This repository, Health, contains two distinct research projects that explore the capabilities of Large Language Models (LLMs) for the detection and mitigation of complex informational and behavioral issues within social contexts (physician visit), alongside the interpretation of imaging findings from diagnostic reports.

## Projects

### 1. EchoProject

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