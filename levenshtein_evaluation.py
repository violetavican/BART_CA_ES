import os
import re
import json
import wandb
import numpy as np
import unicodedata
import Levenshtein
import argparse
from collections import defaultdict
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained('./Thesis/bart_tokenizer/')
tokenizer.model_max_length = 1024
parser = argparse.ArgumentParser(description='Seleccionar directorios según opción')
parser.add_argument('option', choices=['cat', 'es', 'both'], help="Opciones disponibles: 'cat', 'es', 'both'")
parser.add_argument('mode', choices=['curriculum', 'normal'], help="Opciones disponibles: 'curriculum', 'normal'")
parser.add_argument('calculation', choices=['default', 'task', 'all'], help="Opciones disponibles: 'default', 'task', 'all'")
args = parser.parse_args()
print(f"Doing {args.mode} in {args.option}")
if args.option == "cat":
    if args.mode == 'curriculum':
        directories = ["predictions_json_cat"]
        json_file = 'transformations_log_cat.json'
    else:
        directories = ["equiprobable_predictions_json_cat"]
        json_file = 'equiprobable_transformations_log_cat.json'
if args.option == "es":
    if args.mode == 'curriculum':
        directories = ["predictions_json_es"]
        json_file = 'transformations_log_es.json'
    else:
        directories = ["equiprobable_predictions_json_es"]
        json_file = 'equiprobable_transformations_log_es.json'
if args.option == "both":
    if args.mode == 'curriculum':
        directories = ["predictions_json_cat", "predictions_json_es"]
        json_file = 'transformations_log_both.json'
    else:
        directories = ["equiprobable_predictions_json_es", "equiprobable_predictions_json_cat"]
        json_file = 'equiprobable_transformations_log_both.json'

def calculate_levenshtein_distance(preds, labels, tokenizer):
    char_lev_distances = [
        Levenshtein.distance(pred, label) / max(len(pred), len(label))
        for pred, label in zip(preds, labels)
    ]
    avg_char_lev_distance = np.mean(char_lev_distances)

    tokenized_preds = tokenizer(preds, truncation=True, padding=True)["input_ids"]
    tokenized_labels = tokenizer(labels, truncation=True, padding=True)["input_ids"]
    pad_token_id = tokenizer.pad_token_id

    def levenshtein_distance(pred, label):
        pred_filtered = [tok for tok in pred if tok != pad_token_id]
        label_filtered = [tok for tok in label if tok != pad_token_id]
        return Levenshtein.distance(pred_filtered, label_filtered) / max(len(pred_filtered), len(label_filtered))

    token_lev_distances = [
        levenshtein_distance(pred, label)
        for pred, label in zip(tokenized_preds, tokenized_labels)
    ]
    avg_token_lev_distance = round(np.mean(token_lev_distances), 3)
    return {
        'avg_normalized_levenshtein_distance_char': avg_char_lev_distance,
        'avg_normalized_levenshtein_distance_token': avg_token_lev_distance
    }

pat = re.compile(r'[^\w\s]')
def normalize(texts, pattern):
    accents_mapping = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n',
    'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
    'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U', 'Ñ': 'N',
    'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
    }
    def normalize(text):
        # Remove specified punctuation marks and convert ellipses to spaces directly
        text = pattern.sub('', text) # r'[^\w\s]'
        text = ''.join(accents_mapping.get(char, char) for char in text)
        # Convert the entire string to lowercase
        return text.lower()

    # Apply normalization to each text in the specified column of the batch
    return [normalize(text) for text in texts]

def levenshtein_distance_ratio(preds, labels, tokenizer, norm_texts):
    # How many chars did the model denormalized
    char_pred_label = [
        Levenshtein.distance(pred, label) / max(len(pred), len(label))
        for pred, label in zip(preds, labels)
    ]
    # How many chars are needed to denormalize
    char_norm_label = [
        Levenshtein.distance(norm, label) / max(len(norm), len(label))
        for norm, label in zip(norm_texts, labels)
    ]
    avg_char_ratio = np.mean([pred_lab / norm_lab for pred_lab, norm_lab in zip(char_pred_label, char_norm_label)])

    tokenized_preds = tokenizer(preds, truncation=True, padding=True)["input_ids"]
    tokenized_labels = tokenizer(labels, truncation=True, padding=True)["input_ids"]
    tokenized_norm = tokenizer(norm_texts, truncation=True, padding=True)["input_ids"]
    pad_token_id = tokenizer.pad_token_id
    
    def token_ratio(pred, label, norm):
        pred_filtered = [tok for tok in pred if tok != pad_token_id]
        label_filtered = [tok for tok in label if tok != pad_token_id]
        norm_filtered = [tok for tok in norm if tok != pad_token_id]

        tok_pred_label = Levenshtein.distance(pred_filtered, label_filtered) / max(len(pred_filtered), len(label_filtered))
        tok_norm_label = Levenshtein.distance(norm_filtered, label_filtered) / max(len(norm_filtered), len(label_filtered))
        return tok_pred_label/tok_norm_label
    
    tok_ratio = [
        token_ratio(pred, label, norm)
        for pred, label, norm in zip(tokenized_preds, tokenized_labels, tokenized_norm)
    ]
    avg_tok_ratio = round(np.mean(tok_ratio), 3)
    return {
        'avg_char_ratio': avg_char_ratio,
        'avg_tok_ratio': avg_tok_ratio
    }


def normalize_text(text):
    return unicodedata.normalize('NFC', text)

def load_predictions_and_labels(directories):
    all_preds = []
    all_labels = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_preds.extend([normalize_text(pred) for pred in data['predictions']])
                    all_labels.extend([normalize_text(label) for label in data['labels']])
                    print(f"Processing file {filename}")

    return all_preds, all_labels

preds, labels = load_predictions_and_labels(directories)
if args.calculation in ['task', 'all']:
    with open(json_file, 'r') as f:
        transformations_log = json.load(f)
        print()

    task_batches = defaultdict(list)
    for item in transformations_log:
        batch_idx = item['batch_idx']
        for transformation, _ in item['tasks']:
            task_batches[transformation].append(batch_idx)
    print(f"Total number of batches: {len(transformations_log)}")

    print("\nNumber of batches per transformation:")
    for transformation, batch_indices in task_batches.items():
        print(f"{transformation}: applied to {round((len(batch_indices)/len(transformations_log))*100, 2)}% of the batches")

    print(f"Total number of batches: {len(transformations_log)}")

    print("\nNumber of batches per transformation:")
    for transformation, batch_indices in task_batches.items():
        print(f"{transformation}: {len(batch_indices)} batches, representing a {(len(batch_indices)/len(transformations_log))*100}% of application")
        

    levenshtein_char_impact = defaultdict(list)
    levenshtein_token_impact = defaultdict(list)

    levenshtein_char_ratio = defaultdict(list)
    levenshtein_token_ratio = defaultdict(list)

    for transformation, batch_indices in task_batches.items():
        task_preds = []
        task_labels = []

        for batch_idx in batch_indices:
            task_preds.extend([preds[batch_idx]])
            task_labels.extend([labels[batch_idx]])
            
        distances = calculate_levenshtein_distance(task_preds, task_labels, tokenizer)
        levenshtein_char_impact[transformation].append(distances['avg_normalized_levenshtein_distance_char'])
        levenshtein_token_impact[transformation].append(distances['avg_normalized_levenshtein_distance_token'])

        norm_texts = normalize(task_labels, pat)
        ratio = levenshtein_distance_ratio(task_preds, task_labels, tokenizer, norm_texts)
        levenshtein_char_ratio[transformation].append(ratio['avg_char_ratio'])
        levenshtein_token_ratio[transformation].append(ratio['avg_tok_ratio'])

    # Overall average
    avg_levenshtein_char_impact = {transformation: np.mean(values) for transformation, values in levenshtein_char_impact.items()}
    avg_levenshtein_token_impact = {transformation: np.mean(values) for transformation, values in levenshtein_token_impact.items()}

    print("\nImpacto Promedio de las Transformaciones (Levenshtein Char):")
    for transformation, avg_impact in avg_levenshtein_char_impact.items():
        print(f"{transformation}: {round(avg_impact, 6)}")

    print("\nImpacto Promedio de las Transformaciones (Levenshtein Token):")
    for transformation, avg_impact in avg_levenshtein_token_impact.items():
        print(f"{transformation}: {round(avg_impact, 6)}")

    avg_char_ratio = {transformation: np.mean(values) for transformation, values in levenshtein_char_ratio.items()}
    avg_tok_ratio = {transformation: np.mean(values) for transformation, values in levenshtein_token_ratio.items()}


    print("\nRatio Transformations (Levenshtein Char):")
    for transformation, avg_impact in avg_char_ratio.items():
        print(f"{transformation}: {round(avg_impact, 6)}")
    
    print("\nRatio Transformations (Levenshtein Token):")
    for transformation, avg_impact in avg_tok_ratio.items():
        print(f"{transformation}: {round(avg_impact, 6)}")
    
    print("\nPercentage of properly changed chars:")
    for transformation, avg_impact in avg_char_ratio.items():
        print(f"{transformation}: {(1-avg_impact)*100}")

    print("\nPercentage of properly changed tokens:")
    for transformation, avg_impact in avg_tok_ratio.items():
        print(f"{transformation}: {(1-avg_impact)*100}")


if args.calculation in ['default', 'all']:
    metric_result = calculate_levenshtein_distance(preds, labels, tokenizer)
    print(metric_result)
    normalized_texts = normalize(labels, pat)
    ratio_result = levenshtein_distance_ratio(preds, labels, tokenizer, normalized_texts)
    print(f"Raw ratio per char: {round(ratio_result['avg_char_ratio'], 2)}, and per token: {ratio_result['avg_tok_ratio']}")
    print(f"Solved transformation per char: {(1-ratio_result['avg_char_ratio'])*100}, and per token: {(1-ratio_result['avg_tok_ratio'])*100}")
