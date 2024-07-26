from datasets import load_from_disk, concatenate_datasets, load_dataset
from nltk.tokenize import word_tokenize
from transformers import BartConfig, BartForConditionalGeneration, TrainingArguments, DataCollatorForSeq2Seq, Trainer, PreTrainedTokenizerFast

from functools import partial
import numpy as np
import argparse
import math
import wandb
import os

import nltk
if not nltk.find('tokenizers/punkt'):
    nltk.download('punkt')

parser = argparse.ArgumentParser(description='Access token for the DACSA dataset')
parser.add_argument('token', type=str, help="Personal access token for the DACSA dataset")
args = parser.parse_args()
wandb.login()

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

wandb.log({"status": "Loading datasets started"})
# Loading datasets
SEED = 42
pretrain_dataset = load_from_disk("pretrain_ca_es/")
train_dataset = concatenate_datasets([pretrain_dataset, pretrain_dataset])
train_dataset = train_dataset.shuffle(SEED)

spanish = load_dataset("ELiRF/dacsa", "spanish", split='validation', token=args.token)
catalan = load_dataset("ELiRF/dacsa", "catalan", split='validation', token=args.token)
dacsa_dataset = concatenate_datasets([spanish.shuffle(SEED).select(range(15000)), catalan.shuffle(SEED).select(range(15000))])
dacsa_dataset = dacsa_dataset.shuffle(SEED)
wandb.log({"status": "Loading datasets completed"})

wandb.log({"status": "Masking started"})
# Data Preprocessing
def text_infilling_batch(batch, column_name, percentage, random_state):
    def union(tuples):
        max_second_elements = {}
        for first, second in tuples:
            if first in max_second_elements:
                max_second_elements[first] = max(max_second_elements[first], second)
            else:
                max_second_elements[first] = second
                
        # Transform diccionary in a tuple's list
        return [(first, second) for first, second in max_second_elements.items()]
    
    def mask_spans(text, random_state):
        tokens = word_tokenize(text)
        num_to_mask = math.ceil(percentage * len(tokens))
        new_tokens, i = [], 0

        unif_distribution = [round(i) for i in random_state.uniform(0, len(tokens), num_to_mask)]
        span_length = random_state.poisson(3,num_to_mask)
        tuples = list(zip(unif_distribution, span_length))
        sorted_tuples_by_position = sorted(tuples, key=lambda x: x[0])
        
        for (pos, span) in union(sorted_tuples_by_position):
            if num_to_mask > 0:
                if i < pos:
                    new_tokens.extend(tokens[i:pos]) # To deal with things like tokens[5:3]
                    i = pos # Jump to the next tuple position to mask
                
                if i == pos:
                    new_tokens.append('<mask>') # Mask num_to_mask available
                    mask = min(num_to_mask, span)

                    if span > 0:
                        i = pos + mask
                        num_to_mask -= mask
               
                elif i > pos:
                    continue # Next tuple!

            else: # Stop progressing in the loop if num_to_mask is already satisfied
                break
        
        # Add rest of the sentence
        new_tokens.extend(tokens[i:])
        return ' '.join(new_tokens)

    batch["text_infilling"] = [mask_spans(text, random_state) for text in batch[column_name]]
    return batch

random_state = np.random.RandomState(SEED)
partial_train = partial(text_infilling_batch, column_name="text", percentage=0.4, random_state=np.random.RandomState(SEED))
train_dataset = train_dataset.map(partial_train, batched=True, num_proc=10)

partial_eval = partial(text_infilling_batch, column_name="article", percentage=0.4, random_state=np.random.RandomState(SEED))
eval_dataset = dacsa_dataset.map(partial_eval, batched=True, num_proc=10)
wandb.log({"status": "Masking completed"})


# Tokenize the output of the text infilling with the customized BART tokenizer
wandb.log({"status": "Tokenizing started"})
tokenizer = PreTrainedTokenizerFast.from_pretrained('./Thesis/bart_tokenizer/')

def preprocess_function(texts, column_name):
    model_inputs = tokenizer(
        texts['text_infilling'], max_length=1024, truncation=True
    )
    labels = tokenizer(
        texts[column_name], max_length=1024, truncation=True
    )["input_ids"]
    labels = [[(l if l != tokenizer.pad_token else -100) for l in label ] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

train_col = train_dataset.column_names
eval_col = eval_dataset.column_names

train_data = train_dataset.map(partial(preprocess_function, column_name='text'), batched=True, remove_columns=train_col, num_proc=10)
eval_data = eval_dataset.map(partial(preprocess_function, column_name='article'), batched=True, remove_columns=eval_col, num_proc=10)
wandb.log({"status": "Tokenizing completed"})

# 3. Model Initialization
wandb.log({"status": "Training started"})
config = BartConfig.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration(config) 

# 4. Training Configuration
training_args = TrainingArguments(
    output_dir='./results', 
    evaluation_strategy= 'epoch',   # Changed to 24h
    num_train_epochs=3,              
    per_device_train_batch_size=4,  # Values bigger than 4 raise torch.cuda.OutOfMemoryError: CUDA out of memory
    per_device_eval_batch_size=16,  # CUDA out of memory if > 16
    gradient_accumulation_steps=32, # So that the gradients are not uploaded that often thanks to bigger virtual batch
    warmup_ratio=0.1,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=1000,
    save_steps=10000,               # Changed to 6h
    save_total_limit=5,
    fp16=True,
    run_name="BART"
) # Let some steps run, then average steps to predict how long it can take

# Data collator for language modeling
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model = model,
    max_length=1024,
    padding=True,
    pad_to_multiple_of=8
)

# 5. Training Loop
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data, 
    eval_dataset=eval_data,         # eval de Dacsa corpus WITH text_infilling
)

trainer.train()
wandb.log({"status": "Training completed"})


# 6. Evaluation and Saving
wandb.log({"status": "Evaluation started"})
trainer.evaluate()
wandb.log({"status": "Evaluation completed"})
trainer.save_model("./Thesis/final_BART_trained")

# max_steps =  332829 sec
# Number samples 28401375