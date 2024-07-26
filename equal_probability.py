import datasets
from datasets import concatenate_datasets, load_dataset
import numpy as np
import argparse
import math
import wandb
import time
import os
import re
import nltk
if not nltk.find('tokenizers/punkt'):
    nltk.download('punkt')
from transformers import (
    PreTrainedTokenizerFast, BartForConditionalGeneration, 
    TrainingArguments, DataCollatorForSeq2Seq, Trainer, EvalPrediction
)
from finetuningTasks_currLearn import (
    remove_spaces, add_accents, change_accents,
    remove_all_accents, remove_accents, remove_punctuation, 
    join_sentences, lowercase, remove_uppercase,
    normalize, normalize_space_tokenized, 
)
parser = argparse.ArgumentParser(description='Access token for the DACSA dataset')
parser.add_argument('token', type=str, help="Personal access token for the DACSA dataset")
args = parser.parse_args()

wandb.login()

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

wandb.log({"status": "Loading datasets started"})
# Loading datasets
SEED = 42
spanish_t = load_dataset("ELiRF/dacsa", "spanish", split='train', token=args.token)
catalan_t = load_dataset("ELiRF/dacsa", "catalan", split='train', token=args.token)
dacsa_train = concatenate_datasets([spanish_t.shuffle(SEED), catalan_t.shuffle(SEED)])
it_train = dacsa_train.to_iterable_dataset()
print("Train dataset size:", len(dacsa_train))

spanish_e = load_dataset("ELiRF/dacsa", "spanish", split='validation', token=args.token)
catalan_e = load_dataset("ELiRF/dacsa", "catalan", split='validation', token=args.token)
dacsa_eval = concatenate_datasets([spanish_e.shuffle(SEED), catalan_e.shuffle(SEED)])
it_eval = dacsa_eval.to_iterable_dataset()
print("Eval dataset size:", len(dacsa_eval))

wandb.log({"status": "Loading datasets completed"})


class BatchProcessor:
    def __init__(
        self, column: str, tokenizer: PreTrainedTokenizerFast, max_length: int, 
        total_samples: int, num_epochs: int, logging_steps: int=None,
        gradient_accumulation_steps: int=None, per_device_train_batch_size: int=None, 
        per_device_eval_batch_size: int=None, seed: int=42,
    ):
        self.column = column
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_length = max_length
        self.total_samples = total_samples 
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size

        self.rand_state = np.random.RandomState(self.seed)
        self.sample_count = 0  # Contador de muestras procesadas
        self.current_sample = 0
        self.last_active = 10
        self.lowerBound = 0.3
        self.upperBound = 0.85
        self.thresholds = [0] * 11

        self.tasks = [
            self.lowercase_wrapper, self.remove_uppercase_wrapper, self.remove_all_accents_wrapper,
            self.remove_accents_wrapper, self.add_accents_wrapper, self.change_accents_wrapper,
            self.remove_spaces_wrapper, self.normalize_space_tokenized_wrapper, self.join_sentences_wrapper,
            self.remove_punctuation_wrapper, self.normalize_wrapper,
        ]
        self.t_size = len(self.tasks)
        self.pattern = [
            None, re.compile(r'[A-Z]'), None, 
			re.compile(r'[áéíóúüñàèìòùçÁÉÍÓÚÜÑÀÈÌÒÙÇ]'), re.compile(r'[aeiouncAEIOU]'), re.compile(r'[àèòñçÀÈÒáéóÁÉÓ]'),
			re.compile(r' '), None, re.compile(r'(?<!\d)\.(?!\d)|(\.\.\.)'),
		    re.compile(r'[^\w\s]'), re.compile(r'[^\w\s]')
        ]

        if self.logging_steps and self.gradient_accumulation_steps and self.per_device_train_batch_size:
            self.samples_per_logging_step = self.logging_steps * self.gradient_accumulation_steps * self.per_device_train_batch_size
        elif self.logging_steps and self.per_device_eval_batch_size:
            self.samples_per_logging_step = self.logging_steps * self.per_device_eval_batch_size


    def reset_state(self):
        self.rand_state = np.random.RandomState(self.seed)
        self.current_sample = 0  # Reset for each epoch

    def apply_tasks(self, texts):
        processed_text = texts
        percents = np.round(np.fmin(np.fmax(self.rand_state.uniform(size=self.t_size)-self.lowerBound, [0]*self.t_size) * (1/(self.upperBound-self.lowerBound)), [1]*self.t_size), 2)
        if percents[10] > 0:
            # If it is the last taska and the percentage is bigger than 0, then only apply normalize
            processed_text = self.normalize_wrapper(texts=processed_text, pattern=self.pattern[self.last_active])
        else:
            for i, (fn, per) in enumerate(zip(self.tasks, percents)):
                if per > 0:
                    processed_text = fn(texts=processed_text, per=per, pattern=self.pattern[i])
            
        self.sample_count += len(texts)
        
        if self.sample_count % self.samples_per_logging_step == 0:
            wandb.log({"sample_count": self.sample_count})

        return processed_text
    
    def process(self, batch, indices):
        if indices[0] == 0:
            self.reset_state()

        texts = batch[self.column]
        mtexts = self.apply_tasks(texts)

        model_inputs = self.tokenizer(mtexts, max_length=self.max_length, truncation=True)
        labels = self.tokenizer(texts, max_length=self.max_length, truncation=True)["input_ids"]
        labels = [[(l if l != self.tokenizer.pad_token else -100) for l in label ] for label in labels]
        model_inputs["labels"] = labels
        model_inputs["indice"] = indices
        self.current_sample += len(texts)

        return model_inputs
    
    def remove_spaces_wrapper(self, texts, per, pattern):
        return remove_spaces(texts, self.rand_state, per, pattern)
    
    def add_accents_wrapper(self, texts, per, pattern):
        return add_accents(texts, self.rand_state, per, pattern)
    
    def change_accents_wrapper(self, texts, per, pattern):
        return change_accents(texts, self.rand_state, per, pattern)
    
    def remove_all_accents_wrapper(self, texts, per=None, pattern=None):
        return remove_all_accents(texts)
    
    def remove_accents_wrapper(self, texts, per, pattern):
        return remove_accents(texts, self.rand_state, per, pattern)
        
    def remove_punctuation_wrapper(self, texts, per, pattern):
        return remove_punctuation(texts, self.rand_state, per, pattern)
    
    def join_sentences_wrapper(self, texts, per, pattern):
        return join_sentences(texts, self.rand_state, per, pattern)
    
    def lowercase_wrapper(self, texts, per=None, pattern=None):
        return lowercase(texts)
    
    def remove_uppercase_wrapper(self, texts, per, pattern):
        return remove_uppercase(texts, self.rand_state, per, pattern)
    
    def normalize_wrapper(self, texts, per=None, pattern=None):
        return normalize(texts, pattern)
    
    def normalize_space_tokenized_wrapper(self, texts, per=None, pattern=None):
        return normalize_space_tokenized(texts)


tokenizer = PreTrainedTokenizerFast.from_pretrained('./Thesis/bart_tokenizer/')

batch_processor_train = BatchProcessor(
    "article", tokenizer, max_length=1024, seed=SEED, total_samples=len(dacsa_train), 
    num_epochs=3, logging_steps=572, gradient_accumulation_steps=32,
    per_device_train_batch_size=4,
)

# Important to use map's argument 'with_indices'!!!
train_ds = it_train.map(batch_processor_train.process, batched=True, with_indices=True, remove_columns=dacsa_train.column_names)

def get_num_steps(
    len_ds: int, num_epochs: int, per_device_batch_size: int,
    gradient_accumulation_steps: int, num_devices: int
):
    """
        Torna el nombre de steps que 'ocupa' el dataset en total
    """ 
    batch_size = (per_device_batch_size * gradient_accumulation_steps * num_devices)
    steps_per_epoch = math.ceil(len_ds / batch_size)
    return steps_per_epoch * num_epochs

print(
    "max_steps = ",
    get_num_steps(len(dacsa_train), num_epochs=3, per_device_batch_size=4, gradient_accumulation_steps=32, num_devices=1)
)

batch_processor_eval = BatchProcessor(
    "article", tokenizer, max_length=1024, seed=SEED, total_samples=len(dacsa_eval),
    num_epochs=3,  # Optional, just to maintain a consistent API
    logging_steps=572, per_device_eval_batch_size=16,
)
eval_ds = it_eval.map(batch_processor_eval.process, batched=True, with_indices=True, remove_columns=dacsa_eval.column_names)


# 3. Model Initialization
wandb.log({"status": "Training started"})
model = BartForConditionalGeneration.from_pretrained("./BART-ca-es/")

# 4. Training Configuration
training_args = TrainingArguments(
    output_dir='./results_equiprobable',
    evaluation_strategy= 'steps',
    max_steps=get_num_steps(len(dacsa_train), 3, 4, 32, 1), # 57177 steps
    eval_steps=5718,                # 10% max_steps
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Values bigger than 4 raise torch.cuda.OutOfMemoryError: CUDA out of memory
    per_device_eval_batch_size=16,  # CUDA out of memory if > 16
    gradient_accumulation_steps=32, # So that the gradients are not uploaded that often thanks to bigger virtual batch
    warmup_ratio=0.1,                
    weight_decay=0.01,               
    logging_dir='./logs_equiprobable',            
    logging_steps=572,              # 1%
    save_steps=2858,                # 5%
    save_total_limit=5,
    resume_from_checkpoint=True,
    fp16= True,
    split_batches=True,             # the main process will fetch a full batch and slice it into `num_processes` batches for each process.
    run_name="finetuning_equiprobable"
)

# Data collator for language modeling
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model = model,
    padding=True,
    # max_length=1024,              # Already controlled in the BatchProcessor
    # padding='max_length',
    pad_to_multiple_of=8
)

# 5. Training Loop
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds, 
    eval_dataset=eval_ds,           # DACSA eval corpus WITH text_infilling
)

trainer.train()
wandb.log({"status": "Training completed"})


# 6. Evaluation and Saving
wandb.log({"status": "Evaluation started"})
eval_results = trainer.evaluate()
print(eval_results)
wandb.log({"status": "Evaluation completed"})
trainer.save_model("./Thesis/finetuning_equiprobable")

# max_steps = 2439515 st / 2,75 st/sec = 121028,7 sec = 33,6h
# 332829 st -> 33,6h
# x -> 6h; x = 59400 st