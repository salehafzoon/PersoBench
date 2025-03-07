import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from tqdm import tqdm
import random
import numpy as np
import argparse
import os
import warnings
import pandas as pd
from datasets import Dataset

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load the dataset
def load_nli_data(batch_size):
    
    # Load the dataset using pandas
    train_df = pd.read_json('dnli/dnli/dialogue_nli/dialogue_nli_train.jsonl')
    test_df = pd.read_json('dnli/dnli/dialogue_nli/dialogue_nli_test.jsonl')
    
    # Convert to HuggingFace dataset format for tokenization
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset = {'train': train_dataset, 'validation': test_dataset}
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=210)

    # Apply tokenization separately on train and validation datasets
    encoded_train_dataset = dataset['train'].map(tokenize_function, batched=True)
    encoded_test_dataset = dataset['validation'].map(tokenize_function, batched=True)

    # Rename label column and apply the same operations to both datasets
    encoded_train_dataset = encoded_train_dataset.rename_column("label", "labels")
    encoded_test_dataset = encoded_test_dataset.rename_column("label", "labels")

    # Encode string labels to numerical values for both datasets
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    encoded_train_dataset = encoded_train_dataset.map(lambda example: {'labels': label_mapping[example['labels']]})
    encoded_test_dataset = encoded_test_dataset.map(lambda example: {'labels': label_mapping[example['labels']]})

    # Set the format for both datasets
    encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # Return dataloaders for both datasets
    train_dataloader = DataLoader(encoded_train_dataset, sampler=RandomSampler(encoded_train_dataset), batch_size=batch_size)
    dev_dataloader = DataLoader(encoded_test_dataset, sampler=SequentialSampler(encoded_test_dataset), batch_size=batch_size)

    return train_dataloader, dev_dataloader

# Define the training loop
def train_model(model, train_dataloader, dev_dataloader, optimizer, scheduler, num_epochs, device, max_grad_norm, output_dir):
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    model.to(device)

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)

        acc = correct / total
        logging.info(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
           
    # Save the final model (ensure .bin format)
    final_model_path = os.path.join(output_dir, "bert_dnli")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)  # Saves the tokenizer alongside
    
    # Save the model weights to a .bin file
    torch.save(model.state_dict(), os.path.join(final_model_path, 'bert_dnli.bin'))

    logging.info(f"Final model and tokenizer saved to {final_model_path}")


    return best_acc

def parse_args():
    ap = argparse.ArgumentParser("Arguments for BERT NLI fine-tuning")
    ap.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    ap.add_argument('-ep', '--num_epochs', type=int, default=1, help='Number of epochs')
    ap.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    ap.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    ap.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--output_dir', type=str, default='output', help='Directory to save the fine-tuned model')
    ap.add_argument('--model_dir', type=str, default='models', help='Directory to load/save the pretrained BERT model')
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    warnings.filterwarnings("ignore")
    
    # Check if the model is already downloaded
    model_path = os.path.join(args.model_dir, 'bert-base-uncased')
    if not os.path.exists(model_path):
        logging.info(f"Downloading BERT model to {model_path}")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        logging.info(f"Loading BERT model from {model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    train_dataloader, dev_dataloader = load_nli_data(args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    best_acc = train_model(model, train_dataloader, dev_dataloader, optimizer, scheduler, args.num_epochs, args.device, args.max_grad_norm, args.output_dir)

    logging.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")


# running command:
# python bert_on_dnli.py --batch_size 16 --num_epochs 1 --lr 2e-5 --output_dir output --model_dir models 