# Text Denormalization System for Spanish and Catalan

This repository contains the code developed for my Bachelor's thesis, *"Text Denormalization System Based on Neural Models for Spanish and Catalan"*. The thesis focuses on creating a system capable of transforming the output from automatic speech recognition systems into standard written text for both Spanish and Catalan. The final thesis document can be accessed [here](https://drive.google.com/file/d/1hjIaqkOEBcWpmi3lam4CJ96m1GbrQf22/view).

## Project Overview

The goal of this project was to develop a bilingual model using BART (Bidirectional and Auto-Regressive Transformers) to perform text denormalization tasks in Spanish and Catalan. The denormalization tasks included correcting punctuation, capitalization, and accentuation in text that was originally output as lowercase and unpunctuated by speech recognition systems.

Two different fine-tuning techniques were employed to enhance the denormalization performance:

1. **Equal Probability Fine-Tuning**: This technique involved equally likely sampling of different tasks to fine-tune the model's denormalization capabilities.

2. **Curriculum Learning Fine-Tuning**: Inspired by human learning, this technique started with simpler tasks and gradually introduced more complex ones, improving the model's performance progressively.

## Directory Structure

- **BART_pretraining.py**: Script for pretraining the BART model with a bilingual tokenizer to handle both Spanish and Catalan texts.
- **BART_tokenizer.py**: Custom tokenizer script that was further trained on the DACSA corpus to adapt the BART model for Spanish and Catalan.
- **curriculum_learning.py**: Implements the curriculum learning strategy, applying different tasks in a specific order of difficulty to fine-tune the model.
- **equal_probability.py**: Implements the equal probability strategy for task sampling during fine-tuning.
- **finetuningTasks_currLearn.py**: Script that defines the specific tasks used in the curriculum learning approach, such as adding or removing accents, punctuation, etc.
- **levenshtein_evaluation.py**: Script used to evaluate the model's performance using the Levenshtein distance as a metric, comparing the output against the ground truth.
- **model_config.py**: Configuration file that contains all the necessary parameters and hyperparameters used for model training and evaluation.
- **requirements.txt**: Lists all the Python dependencies required to run the project.

## Installation

To run the code, you need to have Python 3.x installed along with the required packages listed in `requirements.txt`. Install the dependencies with:

```bash
pip install -r requirements.txt
```
