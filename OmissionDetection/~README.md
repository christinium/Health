# Omission Detection in Medical Notes
*A project demonstrating LLM application in healthcare quality improvement*

## Clinical Problem

Medical documentation often omits clinically relevant information from visit transcripts. These omissions can impact:
- Clinical decision-making
- Continuity of care
- Patient safety
- Quality metrics

**Research Question:** Can LLMs reliably identify and classify clinically significant omissions in HPI notes?

---

## Solution Overview

This tool uses GPT-4-32k to detect omissions in medical documentation by:
1. Comparing patient visit transcripts with generated HPI notes
2. Identifying clinically relevant missing information
3. Classifying omissions by severity (Critical, Moderate, Optional)
4. Calculating severity scores to quantify documentation quality

---

## My Approach

### Design Decisions

**Why GPT-4-32k?**
- 32k token context handles long transcripts + full HPIs (up to ~32,768 tokens)
- Strong medical reasoning without fine-tuning
- Superior instruction-following for structured CSV outputs
- Strong zero-shot performance

**Two-stage pipeline:**
1. **Identification** - Prevents severity classification from biasing what's detected
2. **Classification** - Separate task allows focused reasoning on clinical impact

**Severity framework:**
- **Critical**: Essential for clinical decision-making or patient safety (e.g., chest pain characteristics, medication allergies)
- **Moderate**: Important context that supports care or patient-physician relationship (e.g., patient concerns about procedure, family history details)
- **Optional**: Additional details that may be helpful but not necessary (e.g., family gathering plans, tangential conversation)

### Prompt Engineering Strategy

**Key principles I applied:**
- **Role-based prompting**: "You are a physician..." leverages domain knowledge
- **Explicit constraints**: "Only information present in transcript" reduces hallucination
- **Sequential processing**: Identify-then-classify reduces cognitive load
- **Structured output**: CSV format enables quantitative analysis
- **Context preservation**: Keeps transcript and note together to maintain clinical context

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

## Validation Approach

### What I Did
- **Ground truth**: Single-clinician evaluation (me)
- **Manual review**: Validated omissions against source transcripts
- **Severity scoring**: Weighted by transcript length to normalize
- **Temperature setting**: 0.7 (balance between creativity and consistency)

### Metrics Considered for future
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

### Known Edge Cases
- Transcripts with significant off-topic conversation (may flag irrelevant content)
- Notes with extensive medical abbreviations (may not match transcript phrasing)
- Encounters where significant portions are non-verbal (physical exam findings)
- Specialties with different documentation norms (psychiatry vs. emergency medicine)

---

## Technical Implementation

### Architecture
```
Input CSV (Transcript, HPI, Encounter ID)
    ↓
IdentifyOmission7() → GPT-4-32k identifies missing clinical info
    ↓
FilterAndFormatFinal() → GPT-4-32k classifies by severity
    ↓
calculate_severity_score() → Quantitative scoring
    ↓
Output: Classified omissions + weighted severity scores
```

### Key Functions

**`IdentifyOmission7(transcript, note)`**
- Identifies clinically relevant information missing from the HPI
- Uses GPT-4-32k model with physician persona
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

## Usage Guide

### Requirements
```
pandas, numpy, matplotlib, openai
```

### Configuration
```python
API_KEY = 'YOUR_API_KEY'
API_VERSION = '2024-06-01'
RESOURCE_ENDPOINT = 'https://unified-api.ucsf.edu/general'
```

### 1. Prepare Input Data
Create CSV with columns:
- `Transcript` - Full visit transcript
- `HPI` - Generated History of Present Illness note
- `Encounter ID` - Unique identifier

### 2. Run Analysis
```python
# Load data
df = pd.read_csv('your_file.csv')

# Process each encounter
for index, row in df.iterrows():
    # Stage 1: Identify omissions
    df.at[index, 'Omissions'] = IdentifyOmission7(row['Transcript'], row['HPI'])
    
    # Stage 2: Classify by severity
    df.at[index, 'Classified'] = FilterAndFormatFinal(row['Omissions'])

# Calculate scores
df['Transcript_Char_Count'] = df['Transcript'].str.len()
df['Severity_Score'] = df['Classified'].apply(calculate_severity_score)
df['Weighted_Severity_Score'] = df['Severity_Score'] / (df['Transcript_Char_Count'] / 10000)
```

### 3. Export Results
```python
# Save aggregate results
df.to_csv('FinalOmissionsGraded.csv', index=True)

# Export individual encounter results
for index, row in df.iterrows():
    if pd.notna(row['Classified']):
        temp_df = pd.read_csv(io.StringIO(row['Classified']), header=0)
        temp_df.to_csv(f"{row['Encounter ID']}.csv", index=True)
```

---

## Reflections & Future Directions

### What I Learned
- Sequential prompt design significantly reduces hallucination risk
- Explicit role-based prompting improves clinical reasoning
- Temperature tuning balances consistency with nuanced classification
- Validation is the bottleneck - technical implementation was straightforward

### Next Steps
1. **Expand validation cohort** - Multi-rater evaluation across specialties
2. **Test temperature=0** - Assess reproducibility vs. clinical appropriateness trade-off
3. **Develop specialty-specific prompts** - Tailor severity criteria to documentation norms
4. **Compare to traditional methods** - Benchmark against manual chart review
5. **Assess training impact** - Could this improve documentation quality through feedback?

### Potential Applications
- **Real-time documentation assistance** - Flag omissions during note writing
- **Quality improvement feedback loops** - Aggregate patterns for targeted training
- **Educational tool** - Help residents understand complete documentation
- **Audit support** - Streamline compliance reviews for billing/quality metrics

---
