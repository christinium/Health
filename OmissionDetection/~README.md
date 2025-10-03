# Omission Detection in Medical Notes

A Python-based tool for identifying and classifying clinically relevant information missing from medical History of Present Illness (HPI) notes by comparing them against patient visit transcripts.

## Overview

This tool uses Large Language Models (LLMs) to detect omissions in medical documentation by:
1. Comparing patient visit transcripts with generated HPI notes
2. Identifying clinically relevant missing information
3. Classifying omissions by severity (Critical, Moderate, Optional)
4. Calculating severity scores to quantify documentation quality

## Key Features

- **Automated Omission Detection**: Identifies missing clinical information from HPI notes
- **Severity Classification**: Categorizes omissions into three levels:
  - **Critical**: Essential for clinical decision-making or patient safety
  - **Moderate**: Important context that supports care but not immediately critical
  - **Optional**: Additional details that may be helpful but not necessary
- **Severity Scoring**: Quantifies documentation completeness with weighted scores
- **Batch Processing**: Handles multiple transcript-note pairs efficiently
- **CSV Export**: Outputs detailed classification results for analysis


### Validation Approach
- **Ground Truth**: How were omissions validated? (e.g., expert physician review, inter-rater reliability)
- **Metrics**: 
  - Precision/Recall of omission detection
  - Cohen's Kappa for severity classification agreement
  - Correlation between severity scores and clinical outcomes (if available)

### Study Design
- Number of encounters analyzed: [X]
- Clinical specialties represented: [list]
- Transcript sources: [e.g., simulated patients, real encounters]
- IRB approval status and ethical considerations
2. Limitations & Model Behavior
markdown## Known Limitations

### Model-Specific Issues
- **Hallucination Risk**: Model may identify "omissions" not actually present in transcript
- **Context Window**: 32k tokens may be insufficient for very long encounters
- **Consistency**: GPT-4 responses are non-deterministic; same input may yield different omissions
- **Domain Specificity**: Model not fine-tuned on medical data; relies on general medical knowledge

### Clinical Limitations
- Cannot assess clinical judgment (what *should* be omitted for brevity)
- No understanding of documentation standards specific to specialties
- Cannot distinguish between truly missing vs. intentionally excluded information
- Binary comparison doesn't account for paraphrasing or summarization

### Mitigation Strategies
- Temperature setting: [What did you use? Default is 1.0]
- Multiple runs and consensus scoring (if implemented)
- Human review of Critical classifications



## Technical Details

### LLM Model
- Model: GPT-4-32k via Azure OpenAI
- Deployment: gpt-4-32k
- API Version: 2024-06-01

### Prompt Engineering Approach

The tool uses a two-stage prompt engineering strategy:

#### Stage 1: Omission Identification (IdentifyOmission7)
- **System Prompt**: "You are a physician who is writing a medical note"
- **User Prompt Structure**:
  - Presents the HPI note first
  - Provides the full transcript
  - Explicitly instructs to:
    - Only include information present in the transcript
    - Avoid making assumptions
    - Focus on clinically relevant details
    - Consider impact on clinical decision-making
    - Consider importance to patient-physician relationship

#### Stage 2: Severity Classification (FilterAndFormatFinal)
- **System Prompt**: "You are a physician tasked with ensuring complete and accurate medical documentation."
- **Classification Criteria**:
  - Critical: Information essential for clinical decision-making or directly impacts patient safety
  - Moderate: Important context or supports patient-physician relationship
  - Optional: Additional details that may be helpful but not necessary

### Output Format Requirements
- Comma-delimited CSV with proper escaping
- Three columns: Omission, Severity, Reasoning
- Complete sentences for omissions
- Concise explanations for severity assignments

Example:
```csv
"Patient reported feeling faint after the prostate biopsy and experienced abdominal discomfort, which was relieved with Tylenol.",Moderate,"This information provides valuable context about the patient's reaction to prior procedures and possible anticipatory concerns but does not critically impact immediate clinical decision-making."
```

### Error Handling
- Empty or malformed CSV responses
- Missing severity classifications
- Parsing errors in classification output

## Implementation Details

### Key Functions

#### `IdentifyOmission7(transcript, note)`
**Purpose**: Identifies clinically relevant information missing from the HPI
- Uses GPT-4-32k model
- Processes transcript and note together in single prompt
- Filters for clinical relevance

#### `FilterAndFormatFinal(omissions)`
**Purpose**: Classifies each omission by severity level
- Uses GPT-4-32k model
- Enforces CSV output format
- Includes classification reasoning

#### `calculate_severity_score(csv_string)`
**Purpose**: Computes numerical severity score
- Returns 0 for empty strings
- Returns -7 for parsing errors

#### `split_transcript_into_segments_df(transcript, segment_size=3000)`
**Purpose**: Handles long transcripts
- Word-boundary aware splitting
- Handles varying transcript lengths

### Token Considerations
- GPT-4-32k supports up to 32,768 tokens
- Average transcript: 5,000-15,000 characters
- Automatic segmentation for longer transcripts

### Prompt Design Principles

1. **Sequential Processing**: Omission identification happens before classification to reduce cognitive load on the model
2. **Role-Based Prompting**: Uses physician persona to leverage domain-specific reasoning
3. **Explicit Constraints**: Clear guidelines prevent hallucination and maintain clinical relevance
4. **Structured Output**: CSV format enables programmatic processing and scoring
5. **Context Preservation**: Keeps transcript and note together to maintain clinical context

### Alternative Approaches Considered

| Approach | Pros | Cons | Why Not Used |
|----------|------|------|--------------|
| Rule-based NLP | Interpretable, fast, deterministic | Requires extensive feature engineering, brittle | Poor generalization to varied documentation styles |
| Traditional ML (BERT) | Lower cost, fine-tunable | Needs large labeled dataset | Insufficient labeled training data for this task |
| GPT-3.5 | Lower cost | Weaker medical reasoning | Testing showed inadequate clinical nuance |
| Open-source (Llama-2) | Privacy, cost | Lower performance on medical tasks | Privacy less critical for research POC |

### Why GPT-4-32k?
- Extended context window handles long transcripts + full HPI (up to 32,768 tokens)
- Superior reasoning for medical content and complex clinical scenarios
- Better instruction-following for structured CSV output
- Strong zero-shot performance without need for fine-tuning

## Limitations & Validation Needs

### Current Limitations

#### Model-Specific:
- **Hallucination Risk**: Model may identify "omissions" not actually present in transcript - requires expert validation
- **Non-deterministic**: Same input may yield different results across runs (temperature > 0)
- **No Fine-tuning**: Relies on general medical knowledge, not trained on institution-specific documentation standards
- **Black Box**: Difficult to understand why specific omissions are identified or classified

#### Clinical:
- Cannot assess clinical judgment about appropriate summarization
- No understanding of specialty-specific documentation norms
- Cannot distinguish truly missing vs. intentionally excluded information
- Binary comparison doesn't account for acceptable paraphrasing
- May not capture implicit clinical reasoning documented elsewhere in note

#### Methodological:
- No ground truth for omission identification
- Severity categories not clinically validated
- Scoring system weights (3/1/0) are arbitrary
- Weighted severity normalization by character count untested

### Validation Requirements

#### Before Clinical Use:
1. **Expert Review**: Clinicians must review sample of identified omissions for:
   - Accuracy (true positive rate)
   - Clinical relevance (are these actually important?)
   - Severity appropriateness (do categories align with clinical judgment?)

2. **Inter-rater Reliability**: 
   - Multiple clinicians independently classify omissions
   - Calculate Cohen's Kappa or Fleiss' Kappa
   - Target: κ > 0.60 for acceptable agreement

3. **Specialty Validation**:
   - Test across different specialties (primary care, cardiology, psychiatry, etc.)
   - Assess if omission patterns and severity differ meaningfully

4. **Prospective Testing**:
   - Apply to new encounters not used in development
   - Measure clinical utility: would knowing these omissions change care?

5. **Reproducibility**:
   - Test with temperature = 0 for deterministic outputs
   - Multiple runs with same input to assess consistency

### Known Edge Cases
- Very long transcripts (>20,000 characters) may exceed context limits
- Transcripts with significant off-topic conversation
- Notes with extensive use of medical abbreviations
- Encounters where significant portions are non-verbal

## Requirements

```python
pandas
numpy
matplotlib
openai  # Azure OpenAI
IPython
```

## Configuration

```python
API_KEY = 'YOUR_API_KEY'
API_VERSION = '2024-06-01'
RESOURCE_ENDPOINT = 'https://unified-api.ucsf.edu/general'
```

## Usage Guide

### 1. Prepare Input Data
Create a CSV with:
- Transcript
- HPI
- Encounter ID

### 2. Run Analysis
```python
# Load and process data
df = pd.read_csv('your_file.csv')

# Identify and classify omissions
for index, row in df.iterrows():
    df.at[index, 'Omissions'] = IdentifyOmission7(row['Transcript'], row['HPI'])
    df.at[index, 'Classified'] = FilterAndFormatFinal(row['Omissions'])

# Calculate severity scores
df['Transcript_Char_Count'] = df['Transcript'].str.len()
df['Severity_Score'] = df['Classified'].apply(calculate_severity_score)
df['Weighted_Severity_Score'] = df['Severity_Score'] / (df['Transcript_Char_Count'] / 10000)
```

### 3. Export Results
```python
# Save results
df.to_csv('FinalOmissionsGraded.csv', index=True)

# Export individual encounter results
for index, row in df.iterrows():
    if pd.notna(row['Classified']):
        temp_df = pd.read_csv(io.StringIO(row['Classified']), header=0)
        temp_df.to_csv(f"{row['Encounter ID']}.csv", index=True)
```