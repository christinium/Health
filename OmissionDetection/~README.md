# Omission Detection in Medical Notes

A Python-based tool for identifying and classifying clinically relevant information missing from medical History of Present Illness (HPI) notes by comparing them against patient visit transcripts.

## Overview

This tool uses Large Language Models (LLMs) to detect omissions in medical documentation by:

- Comparing patient visit transcripts with generated HPI notes
- Identifying clinically relevant missing information
- Classifying omissions by severity (Critical, Moderate, Optional)
- Calculating severity scores to quantify documentation quality

## Key Features

### Automated Omission Detection
- Identifies missing clinical information from HPI notes

### Severity Classification
Categorizes omissions into three levels:
- **Critical**: Essential for clinical decision-making or patient safety
- **Moderate**: Important context that supports care but not immediately critical
- **Optional**: Additional details that may be helpful but not necessary

### Additional Features
- Severity Scoring: Quantifies documentation completeness with weighted scores
- Batch Processing: Handles multiple transcript-note pairs efficiently
- CSV Export: Outputs detailed classification results for analysis

## Requirements

```python
pandas
numpy
matplotlib
openai  # Azure OpenAI
IPython
```

## Configuration

The tool connects to Azure OpenAI services. Configure your API credentials:

```python
API_KEY = 'YOUR_API_KEY'
API_VERSION = '2024-06-01'
RESOURCE_ENDPOINT = 'https://unified-api.ucsf.edu/general'
```

## Usage

### 1. Prepare Input Data
Create a CSV file with the following columns:
- Transcript: Patient visit dialogue
- HPI: Generated History of Present Illness note
- Encounter ID: Unique identifier for each visit

### 2. Run Omission Detection

```python
# Load data
df = pd.read_csv('your_file.csv')

# Identify omissions
for index, row in df.iterrows():
    df.at[index, 'Omissions'] = IdentifyOmission7(row['Transcript'], row['HPI'])

# Classify severity
for index, row in df.iterrows():
    df.at[index, 'Classified'] = FilterAndFormatFinal(row['Omissions'])

# Calculate scores
df['Transcript_Char_Count'] = df['Transcript'].str.len()
df['Severity_Score'] = df['Classified'].apply(calculate_severity_score)
df['Weighted_Severity_Score'] = df['Severity_Score'] / (df['Transcript_Char_Count'] / 10000)
```

### 3. Export Results

```python
# Save full results
df.to_csv('FinalOmissionsGraded.csv', index=True)

# Save individual classifications by encounter
for index, row in df.iterrows():
    if pd.notna(row['Classified']):
        temp_df = pd.read_csv(io.StringIO(row['Classified']), header=0)
        temp_df.to_csv(f"{row['Encounter ID']}.csv", index=True)
```

## Severity Scoring

The tool uses a weighted scoring system:
- Critical omissions: 3 points
- Moderate omissions: 1 point
- Optional omissions: 0 points

The `Weighted_Severity_Score` normalizes by transcript length (per 10,000 characters) to enable fair comparisons across encounters of different lengths.

## Output Format

Classification results are provided in CSV format with three columns:

| Column | Description |
|--------|-------------|
| Omission | Complete sentence describing the missing information |
| Severity | Classification level (Critical/Moderate/Optional) |
| Reasoning | Explanation for the assigned severity |

### Example Output

```csv
Omission,Severity,Reasoning
"Patient reported feeling faint after the prostate biopsy and experienced abdominal discomfort, which was relieved with Tylenol.",Moderate,"This information provides valuable context about the patient's reaction to prior procedures and possible anticipatory concerns but does not critically impact immediate clinical decision-making."
```

## Key Functions

### IdentifyOmission7(transcript, note)
Identifies clinically relevant information missing from the HPI by comparing it with the transcript.

**Parameters:**
- `transcript (str)`: The patient visit dialogue
- `note (str)`: The HPI section to evaluate

**Returns:**
- Bullet-point list of omissions

### FilterAndFormatFinal(omissions)
Classifies each omission by severity level.

**Parameters:**
- `omissions (str)`: List of identified omissions

**Returns:**
- CSV-formatted string with classifications

### calculate_severity_score(csv_string)
Computes numerical severity score from classified omissions.

**Parameters:**
- `csv_string (str)`: CSV-formatted classification results

**Returns:**
- Total severity score (int)

### split_transcript_into_segments_df(transcript, segment_size=3000)
Splits long transcripts into manageable segments for processing.

**Parameters:**
- `transcript (str)`: Input text
- `segment_size (int)`: Target segment size in characters

**Returns:**
- DataFrame with segment_id and segment_text columns

## Model Configuration

The tool uses Azure OpenAI's GPT-4-32k model for analysis. The prompts are designed to:
- Focus on clinically relevant information
- Avoid assumptions not present in the transcript
- Prioritize information affecting patient safety and decision-making
- Use proper CSV formatting with escaping rules