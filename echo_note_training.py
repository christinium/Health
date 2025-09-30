import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import ast

# ============================================================================
# STEP 1: LOAD YOUR ECHO DATA
# ============================================================================

# Load your CSV
df = pd.read_csv('your_echo_data.csv')  # Change to your actual filename

print("Data loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst row:")
print(df.head(1))

# ============================================================================
# STEP 2: PARSE LABELS
# ============================================================================

# Label names in order
LABEL_NAMES = [
    'LA_cavity',
    'RA_dilated', 
    'LV_systolic',
    'LV_cavity',
    'LV_wall',
    'RV_cavity',
    'RV_systolic',
    'AV_stenosis',
    'MV_stenosis',
    'TV_regurgitation',
    'TV_stenosis',
    'TV_pulm_htn',
    'AV_regurgitation',
    'MV_regurgitation',
    'RA_pressure',
    'LV_diastolic',
    'RV_volume_overload',
    'RV_wall',
    'RV_pressure_overload'
]

# If labels are stored as strings like "[1,0,1,...]", parse them
def parse_labels(label_str):
    """Convert string representation of list to actual list."""
    if isinstance(label_str, str):
        return ast.literal_eval(label_str)
    elif isinstance(label_str, list):
        return label_str
    else:
        # If it's already parsed or in another format
        return label_str

# Apply parsing
df['labels_parsed'] = df['labels'].apply(parse_labels)

# Verify labels are correct length
assert all(len(labels) == 19 for labels in df['labels_parsed']), \
    "Not all label arrays have 19 values!"

print("\nLabels parsed successfully!")
print(f"Example labels: {df['labels_parsed'].iloc[0]}")

# ============================================================================
# STEP 3: FORMAT FOR GEMMA
# ============================================================================

def format_echo_prompt(row):
    """
    Format echo report into Gemma instruction format.
    
    The model will predict all 19 values as a structured output.
    """
    input_text = row['input']  # Change 'input' to your actual column name
    labels = row['labels_parsed']
    
    # Create formatted label string
    label_pairs = [f"{LABEL_NAMES[i]}: {labels[i]}" for i in range(19)]
    label_text = "\n".join(label_pairs)
    
    prompt = f"""<start_of_turn>user
Analyze this echocardiogram report and provide assessment values for each cardiac feature. Output should be in the format "feature: value" for each of the 19 features.

Report:
{input_text}<end_of_turn>
<start_of_turn>model
{label_text}<end_of_turn>"""
    
    return prompt

# Apply formatting
df['text'] = df.apply(format_echo_prompt, axis=1)

print("\n" + "="*70)
print("FORMATTED EXAMPLE:")
print("="*70)
print(df['text'].iloc[0])
print("="*70)

# ============================================================================
# STEP 4: DATA CLEANING
# ============================================================================

print(f"\nBefore cleaning: {len(df)} samples")

# Remove any rows with missing data
df = df.dropna(subset=['text'])

# Remove duplicates based on input text
df = df.drop_duplicates(subset=['input'])

# Filter by text length (optional - adjust limits based on your data)
df['text_length'] = df['text'].str.len()
print(f"\nText length stats:")
print(df['text_length'].describe())

# Remove extremely short or long examples if needed
# df = df[(df['text_length'] > 100) & (df['text_length'] < 4096)]

df = df.drop(columns=['text_length'])

print(f"After cleaning: {len(df)} samples")

# ============================================================================
# STEP 5: CREATE TRAIN / TUNE / TEST SPLIT
# ============================================================================

"""
Split strategy for medical data:
- Train: 70% - for model training
- Tune (validation): 15% - for hyperparameter tuning and early stopping
- Test: 15% - held out for final evaluation
"""

# First split: separate test set (15%)
train_tune_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    shuffle=True
)

# Second split: separate train and tune from remaining 85%
# 70% of total = 82.35% of train_tune
# 15% of total = 17.65% of train_tune
train_df, tune_df = train_test_split(
    train_tune_df,
    test_size=0.1765,  # This gives us 15% of original
    random_state=42,
    shuffle=True
)

print("\n" + "="*70)
print("DATASET SPLITS:")
print("="*70)
print(f"Training set:   {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Tuning set:     {len(tune_df)} samples ({len(tune_df)/len(df)*100:.1f}%)")
print(f"Test set:       {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
print(f"Total:          {len(df)} samples")

# ============================================================================
# STEP 6: SAVE SPLITS TO CSV (OPTIONAL BUT RECOMMENDED)
# ============================================================================

# Save splits so you can reload them later without re-splitting
train_df.to_csv('echo_train.csv', index=False)
tune_df.to_csv('echo_tune.csv', index=False)
test_df.to_csv('echo_test.csv', index=False)

print("\n✓ Splits saved to CSV files:")
print("  - echo_train.csv")
print("  - echo_tune.csv")
print("  - echo_test.csv")

# ============================================================================
# STEP 7: CONVERT TO HUGGING FACE DATASET FORMAT
# ============================================================================

# Create datasets from the DataFrames
train_dataset = Dataset.from_pandas(train_df[['text']], preserve_index=False)
tune_dataset = Dataset.from_pandas(tune_df[['text']], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[['text']], preserve_index=False)

# Combine into a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': tune_dataset,  # This is your "tune" set
    'test': test_dataset
})

print("\n" + "="*70)
print("HUGGING FACE DATASET:")
print("="*70)
print(dataset)

# ============================================================================
# STEP 8: TOKENIZE THE DATASETS
# ============================================================================

def tokenize_function(examples):
    """Tokenize the text data."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,  # Longer for medical reports
        padding="max_length",
    )

# Tokenize all splits
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing datasets",
)

print("\n" + "="*70)
print("TOKENIZED DATASETS:")
print("="*70)
print(tokenized_datasets)

# ============================================================================
# STEP 9: CONFIGURE TRAINER FOR MEDICAL DATA
# ============================================================================

from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments optimized for your dataset
training_args = TrainingArguments(
    output_dir="./gemma_echo_finetuned",
    
    # Epochs - adjust based on your data size
    num_train_epochs=5,  # Start with 5, adjust based on validation loss
    
    # Batch sizes
    per_device_train_batch_size=2,  # Smaller for longer medical texts
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    
    # Learning rate
    learning_rate=2e-5,  # Conservative for medical domain
    weight_decay=0.01,
    
    # Precision
    fp16=False,
    bf16=True,
    
    # Logging
    logging_dir='./logs',
    logging_steps=50,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=100,
    
    # Saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,  # Keep only best 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Early stopping - will stop if no improvement
    warmup_steps=100,
    lr_scheduler_type="cosine",
    
    # Optimizer
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    
    report_to="none",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # Your "tune" set
    data_collator=data_collator,
)

print("\n✓ Trainer configured for echo report data!")
print(f"Training on:   {len(tokenized_datasets['train'])} samples")
print(f"Validating on: {len(tokenized_datasets['validation'])} samples")
print(f"Test set:      {len(tokenized_datasets['test'])} samples (held out)")

# ============================================================================
# STEP 10: START TRAINING
# ============================================================================

print("\n" + "="*70)
print("READY TO TRAIN!")
print("="*70)
print("Run: trainer.train()")
print("\nAfter training, evaluate on test set with:")
print("test_results = trainer.evaluate(tokenized_datasets['test'])")

# ============================================================================
# EVALUATION HELPER FOR TEST SET
# ============================================================================

def evaluate_on_test_set():
    """
    Evaluate the fine-tuned model on the held-out test set.
    Run this AFTER training is complete.
    """
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    test_results = trainer.evaluate(tokenized_datasets['test'])
    
    print("\nTest Set Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    return test_results

# ============================================================================
# NOTES FOR MEDICAL DATA
# ============================================================================

"""
IMPORTANT CONSIDERATIONS FOR MEDICAL ECHO DATA:

1. DATA BALANCE:
   - Check if your 19 labels are balanced
   - Medical data often has class imbalance
   - Consider stratified splitting if certain conditions are rare

2. EVALUATION METRICS:
   - Loss alone may not be sufficient
   - Consider implementing custom metrics (F1, precision, recall per label)
   - Medical predictions need high precision

3. MODEL SIZE:
   - Gemma 2B is good for prototyping
   - Consider Gemma 9B for production if accuracy is critical
   
4. TRAINING TIPS:
   - Monitor validation loss closely
   - Stop if validation loss stops decreasing (early stopping)
   - Medical domain may need more epochs (5-10)
   - Lower learning rate is safer for specialized domains

5. PROMPT ENGINEERING:
   - Current format has model output all 19 values at once
   - Alternative: Ask for one value at a time (more reliable but slower)
   - Consider adding medical context in the prompt

6. DATA AUGMENTATION (optional):
   - Paraphrase reports while keeping medical meaning
   - Add synthetic examples for rare conditions
   - Mix with general medical QA data

7. TESTING:
   - The test set is completely held out - don't touch it until final evaluation
   - Use tune/validation set for all hyperparameter decisions
   - Only evaluate on test set once at the very end
"""