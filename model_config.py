import argparse
from transformers import BartForConditionalGeneration

# Set up argument parser
parser = argparse.ArgumentParser(description="Load model configuration.")
parser.add_argument("model_name", type=str, help="Name of the model to load. Options: 'curriculum', 'equal_probability'")

# Parse the arguments
args = parser.parse_args()

# Determine the model path and output file name based on the input argument
if args.model_name == "curriculum":
    model_path = './Thesis/finetuning_currLearning'
    output_file_name = "curriculum_model_config.txt"
elif args.model_name == "equal_probability":
    model_path = './Thesis/finetuning_equiprobable'
    output_file_name = "equal_probability_model_config.txt"
else:
    raise ValueError("Invalid model name. Choose 'curriculum' or 'equal_probability'.")

# Load the model
model = BartForConditionalGeneration.from_pretrained(model_path)

# Load the model configuration
config = model.config

# Open a text file to write the output with the dynamic name
with open(output_file_name, "w") as file:
    # Write configuration parameters to the file
    file.write("Model Configuration Parameters:\n")
    file.write(f"max_length: {config.max_length}\n")
    file.write(f"max_position_embeddings: {config.max_position_embeddings}\n")
    file.write(f"vocab_size: {config.vocab_size}\n")
    file.write(f"d_model: {config.d_model}\n")
    file.write(f"encoder_layers: {config.encoder_layers}\n")
    file.write(f"decoder_layers: {config.decoder_layers}\n")

    # Write default generation parameters to the file
    file.write("\nDefault Generation Parameters:\n")
    file.write(f"max_length: {config.max_length}\n")
    file.write(f"min_length: {config.min_length}\n")
    file.write(f"do_sample: {config.do_sample}\n")
    file.write(f"num_beams: {config.num_beams}\n")
    file.write(f"temperature: {config.temperature}\n")
    file.write(f"top_k: {config.top_k}\n")
    file.write(f"top_p: {config.top_p}\n")
    file.write(f"repetition_penalty: {config.repetition_penalty}\n")
