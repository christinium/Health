# Omission Detection in Medical Notes
*A project demonstrating LLM application in healthcare quality improvement*

## TL;DR

Traditional NLP metrics (BLEU/ROUGE) don't work for clinical workflows. This system uses a clinical safety-first approach with zero-shot learning and chain-of-thought prompting to detect omissions in medical text—achieving **sub-human error rates**.

**Goal**: Improve the quality and completeness of medical documentation  
**Objective**: Automatically detect clinically relevant details from transcripts that are missing from the HPI section of medical notes

---

## Clinical Problem

Medical documentation often omits clinically relevant information from visit transcripts. These omissions can impact:
- Clinical decision-making
- Continuity of care
- Patient safety
- Quality metrics

**Research Question:** Can LLMs reliably identify and classify clinically significant omissions in HPI notes?

---

## Key Innovation

Traditional NLP evaluation metrics like BLEU and ROUGE scores focus primarily on surface-level textual similarity. However, **these metrics are insufficient for clinical workflows** where understanding semantic nuances, contextual dependencies, and domain-specific medical knowledge is paramount for patient safety and effective decision-making.

By focusing on **clinical impact rather than linguistic similarity**, this framework ensures that:
- Critical medical information is never omitted
- Generated summaries are clinically safe and actionable
- Patient care quality is maintained or improved
- Administrative burden on clinicians is reduced without compromising safety

---

## Solution Overview

This tool uses GPT-4 to detect omissions in medical documentation by:
1. Comparing patient visit transcripts with generated HPI notes
2. Identifying clinically relevant missing information
3. Classifying omissions by severity (Critical, Moderate, Optional)
4. Calculating severity scores to quantify documentation quality

---

## Defining Omissions

An **omission** is the absence of important information that should have been included in the patient's medical record. 

Missing information is considered an omission only if it meets at least one of these criteria:
- **Clinically relevant and necessary** for a complete and accurate understanding of the patient's condition
- **Potentially impactful** on clinical decision-making and patient care
- **Beneficial** for enhancing the patient-physician relationship

**Important Note**: If missing information does NOT fit the criteria above, it is NOT considered an omission.

---

## Three-Tiered Risk Classification

Omissions are categorized based on their clinical impact:

### Critical
- Missing information that significantly impacts clinical decision-making or patient safety
- These omissions could lead to diagnostic errors or inappropriate treatment decisions
- Examples: Chest pain characteristics, medication allergies

### Moderate
- Missing details that provide important context or supplemental information but are not critical to immediate decision-making
- Includes information important for the physician-patient relationship, such as factors that affect trust, communication, and understanding
- Examples: Patient concerns about procedure, not documenting patient treatment preferences

### Optional
- Missing information that is useful but not necessary for clinical decision-making
- Minor nuances or additional context that do not significantly impact understanding of the patient's condition
- Examples: Family gathering plans, tangential conversation

---

## Technical Approach

### Design Decisions

**Why GPT-4?**
- 32k token context handles long transcripts + full HPIs
- Strong medical reasoning without fine-tuning (zero-shot learning)
- Superior instruction-following for structured CSV outputs
- Chain-of-thought prompting enhances reasoning and reduces errors

**Two-stage pipeline:**
1. **Identification** - Prevents severity classification from biasing what's detected
2. **Classification** - Separate task allows focused reasoning on clinical impact

### Pipeline Summary

1. **Extract** omitted content using GPT-4
2. **Classify** omissions by clinical importance (Critical, Moderate, Optional)
3. **Score** and format outputs into structured CSVs

### Prompt Engineering Strategy

**Key principles applied:**
- **Role-based prompting**: "You are a physician..." leverages domain knowledge
- **Explicit constraints**: "Only information present in transcript" reduces hallucination
- **Sequential processing**: Identify-then-classify reduces cognitive load
- **Structured output**: CSV format enables quantitative analysis
- **Context preservation**: Keeps transcript and note together to maintain clinical context
- **Clinician-in-the-loop validation**: Ensures clinical accuracy through expert review

#### Stage 1: Omission Identification
**System Prompt:** "You are a physician who is writing a medical note"

**User Prompt instructs the model to:**
- Present HPI note first, then full transcript
- Only include information present in the transcript
- Avoid making assumptions
- Focus on clinically relevant details
- Consider impact on clinical decision-making and patient-physician relationship

#### Stage 2: Severity Classification
**System Prompt:** "You are a physician tasked with ensuring complete and accurate medical documentation."

**Classification criteria with reasoning:**
- Each omission gets: Description, Severity level, Clinical reasoning
- Output as comma-delimited CSV with proper escaping

**Example output:**
```csv
"Patient reported feeling faint after the prostate biopsy and experienced abdominal discomfort, which was relieved with Tylenol.",Moderate,"This information provides valuable context about the patient's reaction to prior procedures and possible anticipatory concerns but does not critically impact immediate clinical decision-making."
```

---

## Evaluation Metrics

The system employs multiple metrics to assess omission rates:

### Basic Metrics
1. **Errors per Note**: Direct measure of the average number of omissions in each note
2. **Errors per Length of Transcript**: Normalizes omissions by transcript length for fair comparisons

### Weighted Metrics

Weighted scoring prioritizes more serious omissions:
- Critical Errors: **2 points**
- Moderate Errors: **1 point**
- Optional Errors: **0 points**

*Example*: A note with 2 critical errors, 1 moderate error, and 0 optional errors =  
(2 × 2) + (1 × 1) + (0 × 0) = **5 weighted points**

3. **Weighted Errors per Note**: Applies severity weighting to errors per note
4. **Weighted Errors per Length of Transcript**: Weighted errors normalized per 10,000 words

---

## Validation Approach

### Current Status: Proof of Concept

This system demonstrates the viability of using clinically-informed metrics for medical documentation evaluation.

### What Was Done
- **Ground truth**: Single-clinician evaluation
- **Manual review**: Validated omissions against source transcripts
- **Severity scoring**: Weighted by transcript length to normalize
- **Temperature setting**: 0.7 (balance between creativity and consistency)

### Next Steps
1. **Improve prompts** by adding additional examples to enhance accuracy
2. **Clinical validation** with a panel of clinicians to verify real-world applicability

### Metrics Considered for Future Validation
- Precision/Recall of omission detection
- Cohen's Kappa for severity classification agreement
- Correlation between severity scores and clinical outcomes

### What's Needed Before Clinical Use

| Validation Need | Purpose | Implementation |
|----------------|---------|----------------|
| **Multi-rater validation** | Establish inter-rater reliability | 3+ clinicians independently classify sample; measure Cohen's kappa; reconcile disagreements |
| **Specialty testing** | Assess documentation norms | Test across primary care, cardiology, psychiatry; adjust severity criteria if needed |
| **Prospective validation** | Measure clinical utility | Apply to new encounters; assess if omissions would change care; compare to chart review audits |
| **Reproducibility testing** | Establish consistency | Set temperature=0; multiple runs on same input; document failure modes |

---

## Limitations & Mitigations

| Limitation | Clinical Impact | Mitigation Strategy |
|-----------|----------------|---------------------|
| **Hallucination risk** | May report "omissions" not in transcript | Human review of all Critical classifications; validate against source |
| **Non-deterministic outputs** | Same input → different results (temp > 0) | Use temperature=0.7 for balance; run multiple times for consensus on ambiguous cases |
| **No clinical panel judgment** | Can't assess appropriate summarization | Expert review required; cannot replace human panel judgment |
| **Context window limits** | Very long encounters (>20k chars) may truncate | Word-boundary aware segmentation at ~3000 characters |
| **No fine-tuning** | Relies on general medical knowledge | Not trained on institution-specific documentation standards |
| **Black box reasoning** | Difficult to understand why omissions identified | Include reasoning field in output; manual review of edge cases |

---

## Technical Implementation

### Architecture
```
Input CSV (Transcript, HPI, Encounter ID)
    ↓
IdentifyOmission7() → GPT-4 identifies missing clinical info
    ↓
FilterAndFormatFinal() → GPT-4 classifies by severity
    ↓
calculate_severity_score() → Quantitative scoring
    ↓
Output: Classified omissions + weighted severity scores
```

### Key Functions

**`IdentifyOmission7(transcript, note)`**
- Identifies clinically relevant information missing from the HPI
- Uses GPT-4 model with physician persona
- Processes transcript and note together in single prompt
- Filters for clinical relevance based on decision-making impact

**`FilterAndFormatFinal(omissions)`**
- Classifies each omission by severity level
- Enforces CSV output format (Omission, Severity, Reasoning)
- Includes clinical reasoning for classification
- Handles empty or malformed responses

**`calculate_severity_score(csv_string)`**
- Computes numerical severity score (Critical=3, Moderate=2, Optional=1)
- Returns 0 for empty strings
- Returns -7 for parsing errors (flags for manual review)

**`split_transcript_into_segments_df(transcript, segment_size=3000)`**
- Handles transcripts exceeding context window
- Word-boundary aware splitting (preserves clinical context)
- Configurable segment size based on typical transcript length

### Error Handling
- Empty or malformed CSV responses → score of -7
- Missing severity classifications → flagged for review
- Parsing errors → logged with encounter ID

---

## Reference

For more details on the clinical safety evaluation framework: [A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation](https://www.nature.com/articles/s41746-025-01670-7) - *npj Digital Medicine* (2025)
