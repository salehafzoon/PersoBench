import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import argparse
import os
import warnings

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
    dataset = load_dataset('snli')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(example):
        return tokenizer(example['premise'], example['hypothesis'], truncation=True, padding='max_length', max_length=128)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")

    # Filter out invalid labels
    encoded_dataset = encoded_dataset.filter(lambda example: example['labels'] in [0, 1, 2])
    
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_dataset = encoded_dataset['train']
    dev_dataset = encoded_dataset['validation']
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)

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
            
            
    # Save the final model
    final_model_path = os.path.join(output_dir, "bert_snli")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)  # Saves the tokenizer alongside
    
    # Save the model weights to a .bin file
    torch.save(model.state_dict(), os.path.join(final_model_path, 'bert_snli.bin'))


    logging.info(f"Final model and tokenizer saved to {final_model_path}")

    return best_acc

def parse_args():
    ap = argparse.ArgumentParser("Arguments for BERT NLI fine-tuning")
    ap.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
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
# python bert_on_snli.py --batch_size 16 --num_epochs 1 --lr 2e-5 --output_dir output --model_dir models 