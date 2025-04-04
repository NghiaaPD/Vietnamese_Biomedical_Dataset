
import os
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
import nltk
nltk.download('punkt')

MODEL_CHECKPOINT = "VietAI/vit5-base"
OUTPUT_DIR = "./models/VietmedSumT5"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 500
FP16 = True  

dataset = load_dataset("leduckhai/VietMed-Sum")

print(dataset)

train_data = dataset["train.train_whole"]  
val_data = dataset["dev.dev_whole"]
test_data = dataset["test.test_whole"]

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    inputs = examples["transcript"]
    targets = examples["summary"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_INPUT_LENGTH, 
        padding="max_length", 
        truncation=True
    )

    labels = tokenizer(
        targets, 
        max_length=MAX_TARGET_LENGTH, 
        padding="max_length", 
        truncation=True
    ).input_ids

    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels
    ]
    
    return model_inputs

tokenized_train = train_data.map(
    preprocess_function, 
    batched=True,
    remove_columns=train_data.column_names
)

tokenized_val = val_data.map(
    preprocess_function, 
    batched=True,
    remove_columns=val_data.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=MODEL_CHECKPOINT,
    padding=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = rouge_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    result = {k: round(v * 100, 4) for k, v in result.items()}
    
    return result

# Thiết lập các tham số huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=3,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=WARMUP_STEPS,
    fp16=FP16,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
)

# Tạo trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

tokenized_test = test_data.map(
    preprocess_function, 
    batched=True,
    remove_columns=test_data.column_names
)

test_results = trainer.evaluate(tokenized_test)
print(f"Test results: {test_results}")

