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