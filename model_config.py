from transformers import BartForConditionalGeneration

# Load the model
model = BartForConditionalGeneration.from_pretrained('./Thesis/finetuning_currLearning')

# Load the model configuration
config = model.config

# Print out configuration parameters
print("Model Configuration Parameters:")
print(f"max_length: {config.max_length}")
print(f"max_position_embeddings: {config.max_position_embeddings}")
print(f"vocab_size: {config.vocab_size}")
print(f"d_model: {config.d_model}")
print(f"encoder_layers: {config.encoder_layers}")
print(f"decoder_layers: {config.decoder_layers}")

# Access default generation parameters
print("\nDefault Generation Parameters:")
print(f"max_length: {config.max_length}")
print(f"min_length: {config.min_length}")
print(f"do_sample: {config.do_sample}")
print(f"num_beams: {config.num_beams}")
print(f"temperature: {config.temperature}")
print(f"top_k: {config.top_k}")
print(f"top_p: {config.top_p}")
print(f"repetition_penalty: {config.repetition_penalty}")
