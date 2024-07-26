from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, trainers
from datasets import load_from_disk, load_dataset
from transformers.utils.versions import require_version


require_version(
    "datasets>=2.11.0",
)


def batch_iterator(
    dataset, batch_length=100000, input_sentence_size=None
):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
        
        for i in range(0, input_sentence_size, batch_length):
            yield dataset[i: i + batch_length]["text"]


train_dataset = load_from_disk("pretrain_ca_es/")

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# Convert to tokenizers library format for training
tk_tokenizer = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
trainer = trainers.BpeTrainer(vocab_size=50265, min_frequency=2, special_tokens=special_tokens)  # vocab_size based on BART's vocab_size
tk_tokenizer.train_from_iterator(batch_iterator(train_dataset), trainer=trainer) 

special_tokens_map =  {
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
    'sep_token': '</s>',
    'pad_token': '<pad>',
    'cls_token': '<s>',
    'mask_token': '<mask>'
}

# Convert back to Hugging Face format and save
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer)
hf_tokenizer.add_special_tokens(special_tokens_map)
# print(hf_tokenizer.special_tokens_map)
hf_tokenizer.save_pretrained("new_bart_tokenizer")
