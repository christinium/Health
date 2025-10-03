pandas
numpy
matplotlib
openai  # Azure OpenAI
IPython
Configuration
The tool connects to Azure OpenAI services. Configure your API credentials:
pythonAPI_KEY = 'YOUR_API_KEY'
API_VERSION = '2024-06-01'
RESOURCE_ENDPOINT = 'https://unified-api.ucsf.edu/general'
Technical Details
LLM Model

Model: GPT-4-32k via Azure OpenAI
Deployment: gpt-4-32k
API Version: 2024-06-01

Prompt Engineering Approach
The tool uses a two-stage prompt engineering strategy:
Stage 1: Omission Identification (IdentifyOmission7)
System Prompt:
You are a physician who is writing a medical note
User Prompt Structure:

Presents the HPI note first
Provides the full transcript
Explicitly instructs to:

Only include information present in the transcript
Avoid making assumptions
Focus on clinically relevant details
Consider impact on clinical decision-making
Consider importance to patient-physician relationship



Key Constraints:

Only identifies omissions that are clinically relevant
Must be potentially impactful on clinical decision-making OR patient care
Must strengthen the patient-physician relationship
No additions beyond what's explicitly stated in the transcript

Stage 2: Severity Classification (FilterAndFormatFinal)
System Prompt:
You are a physician tasked with ensuring complete and accurate medical documentation.
Classification Criteria:

Critical: Information essential for clinical decision-making or directly impacts patient safety
Moderate: Important context or supports patient-physician relationship but doesn't critically affect immediate decision-making
Optional: Additional details that may be helpful but not necessary for understanding the patient's condition

Output Format Requirements:

Comma-delimited CSV with proper escaping
Three columns: Omission, Severity, Reasoning
Complete sentences for omissions (not vague fragments)
Concise explanations for severity assignments

Example of Good vs. Poor Formatting:
✓ Good: "The patient reported feeling faint after the prostate biopsy and experienced abdominal discomfort, which was relieved with Tylenol."

✗ Poor: "Chest discomfort description clarified"
Prompt Design Principles

Sequential Processing: Omission identification happens before classification to reduce cognitive load on the model
Role-Based Prompting: Uses physician persona to leverage domain-specific reasoning
Explicit Constraints: Clear guidelines prevent hallucination and maintain clinical relevance
Structured Output: CSV format enables programmatic processing and scoring
Context Preservation: Keeps transcript and note together to maintain clinical context

Error Handling
The tool includes robust error handling for:

Empty or malformed CSV responses
Missing severity classifications
Parsing errors in classification output

Usage
1. Prepare Input Data
Create a CSV file with the following columns:

Transcript: Patient visit dialogue
HPI: Generated History of Present Illness note
Encounter ID: Unique identifier for each visit

2. Run Omission Detection
python# Load data
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
3. Export Results
python# Save full results
df.to_csv('FinalOmissionsGraded.csv', index=True)

# Save individual classifications by encounter
for index, row in df.iterrows():
    if pd.notna(row['Classified']):
        temp_df = pd.read_csv(io.StringIO(row['Classified']), header=0)
        temp_df.to_csv(f"{row['Encounter ID']}.csv", index=True)
Severity Scoring
The tool uses a weighted scoring system:

Critical omissions: 3 points
Moderate omissions: 1 point
Optional omissions: 0 points

The Weighted_Severity_Score normalizes by transcript length (per 10,000 characters) to enable fair comparisons across encounters of different lengths.
Output Format
Classification results are provided in CSV format with three columns:
ColumnDescriptionOmissionComplete sentence describing the missing informationSeverityClassification level (Critical/Moderate/Optional)ReasoningExplanation for the assigned severity
Example Output
csvOmission,Severity,Reasoning
"Patient reported feeling faint after the prostate biopsy and experienced abdominal discomfort, which was relieved with Tylenol.",Moderate,"This information provides valuable context about the patient's reaction to prior procedures and possible anticipatory concerns but does not critically impact immediate clinical decision-making."
Key Functions
IdentifyOmission7(transcript, note)
Identifies clinically relevant information missing from the HPI by comparing it with the transcript.
Parameters:

transcript (str): The patient visit dialogue
note (str): The HPI section to evaluate

Returns:

Bullet-point list of omissions

Implementation Details:

Uses GPT-4-32k model
Processes transcript and note together in single prompt
Filters for clinical relevance before returning results

FilterAndFormatFinal(omissions)
Classifies each omission by severity level.
Parameters:

omissions (str): List of identified omissions

Returns:

CSV-formatted string with classifications

Implementation Details:

Uses GPT-4-32k model
Enforces CSV output format with proper escaping
Includes reasoning for each classification

calculate_severity_score(csv_string)
Computes numerical severity score from classified omissions.
Parameters:

csv_string (str): CSV-formatted classification results

Returns:

Total severity score (int)

Error Handling:

Returns 0 for empty strings
Returns -7 for parsing errors

split_transcript_into_segments_df(transcript, segment_size=3000)
Splits long transcripts into manageable segments for processing.
Parameters:

transcript (str): Input text
segment_size (int): Target segment size in characters (default: 3000)

Returns:

DataFrame with segment_id and segment_text columns

Features:

Word-boundary aware splitting (doesn't break mid-word)
Handles transcripts shorter than segment_size

Model Configuration
The tool uses Azure OpenAI's GPT-4-32k model for analysis. The prompts are designed to:

Focus on clinically relevant information
Avoid assumptions not present in the transcript
Prioritize information affecting patient safety and decision-making
Use proper CSV formatting with escaping rules

Token Considerations

GPT-4-32k supports up to 32,768 tokens
Average transcript: 5,000-15,000 characters
Transcripts are segmented if needed to stay within limits