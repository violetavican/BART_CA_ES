import re
import math
from typing import List
import numpy as np
import nltk
from nltk import word_tokenize
if not nltk.find('tokenizers/punkt'):
    nltk.download('punkt')

def remove_spaces(texts, random_state, per, pattern, minProb=0, maxProb=0.5):
    def remove_some_spaces(text):
        percentage = per * (maxProb - minProb)
        
        # Calculate the number of spaces to remove
        space_positions = [m.start() for m in pattern.finditer(text)] # r' ', 
        num_to_remove = int(percentage * len(space_positions))
        # Randomly choose spaces to remove
        if num_to_remove > 0:
            remove_positions = set(random_state.choice(space_positions, size=num_to_remove, replace=False))
        else:
            remove_positions = set()
        
        # Create new text by skipping the spaces to be removed
        new_text = ''.join([text[i] for i in range(len(text)) if i not in remove_positions])
        
        return new_text
    return [remove_some_spaces(text) for text in texts]

def add_accents(texts, random_state, per, pattern, minProb=0, maxProb=0.6):
    inverse_accents_mapping = {
        'a': ['à', 'á', 'á', 'á'],
        'e': ['è', 'é', 'é', 'é'],
        'i': ['í'],
        'o': ['ò', 'ó', 'ó', 'ó'],
        'u': ['ú'],
        'n': ['n', 'n', 'n', 'n', 'ñ'],
        'c': ['c', 'c', 'c', 'c', 'ç'],
        'A': ['À', 'Á', 'Á', 'Á'],
        'E': ['È', 'É', 'É', 'É'],
        'I': ['Í'],
        'O': ['Ò', 'Ó', 'Ó', 'Ó'],
        'U': ['Ú'],
    }
    accented_vowels = set('àáèéíòóúñçÀÁÈÉÍÒÓÚ')
    def add_accents(text):
        words = text.split()
        eligible_words = [idx for idx, word in enumerate(words) if not any(char in accented_vowels for char in word)]
        num_eligible = len(eligible_words)

        if num_eligible > 0:
            accent_prob = per * (maxProb - minProb)
            num_to_accent = max(1, math.ceil(accent_prob * num_eligible))
            words_to_accent = set(random_state.choice(eligible_words, size=num_to_accent, replace=False))

            new_words = []
            for idx, word in enumerate(words):
                if idx in words_to_accent:
                    vowel_positions = [m.start() for m in pattern.finditer(word)] # r'[aeiouncAEIOU]',
                    if vowel_positions:
                        # Randomly select one vowel to accent
                        accent_position = random_state.choice(vowel_positions)
                        accented_vowel = random_state.choice(inverse_accents_mapping[word[accent_position]])
                        # Rebuild the word with the accented vowel
                        word = word[:accent_position] + accented_vowel + word[accent_position+1:]
                new_words.append(word)
            return ' '.join(new_words)
        else:
            return text  # No changes if no eligible words
    return [add_accents(text) for text in texts]

def change_accents(texts, random_state, per, pattern, minProb=0, maxProb=0.75):
    # Mapping from unaccented to possible accented characters
    inverse_accents_mapping = {
        'à': ['á'],
        'è': ['é'],
        'ò': ['ó'],
        'ñ': ['n'],
        'ç': ['c'],
        'À': ['Á'],
        'È': ['É'],
        'Ò': ['Ó'],
        'á': ['à'],
        'é': ['è'],
        'ó': ['ò'],
        'Á': ['À'],
        'É': ['È'],
        'Ó': ['Ò'],
    }

    def change_accents(text):
        # Find positions of all vowels that could potentially receive an accent
        percentage = per * (maxProb - minProb)
        vowel_positions = [m.start() for m in pattern.finditer(text)] # r'[àèòñçÀÈÒáéóÁÉÓ]', 
        num_to_add = math.ceil(percentage * len(vowel_positions))
        if num_to_add == 0:
            return text
        
        add_indices = set(random_state.choice(vowel_positions, size=num_to_add, replace=False))
        new_text = []
        last_index = 0
        for pos in vowel_positions:
            if pos in add_indices:
                # Append the text up to the vowel
                new_text.append(text[last_index:pos])
                # Randomly choose an accented form for the vowel to add
                vowel = text[pos]
                accented_vowel = random_state.choice(inverse_accents_mapping[vowel])
                new_text.append(accented_vowel)
            else:
                # If not adding an accent, append text segment including the current character
                new_text.append(text[last_index:pos+1])
            last_index = pos + 1
        new_text.append(text[last_index:])  # Add the remainder of the text
        return ''.join(new_text)

    # Apply the function to add accents to each text in the specified column of the batch
    return [change_accents(text) for text in texts]

def remove_all_accents(texts):
    translation_table = str.maketrans({
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u', 'ç': 'c',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U', 'Ñ': 'N',
        'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U', 'Ç': 'C'
    })
    return [text.translate(translation_table) for text in texts]

def remove_accents(texts, random_state, per, pattern, minProb=0, maxProb=1.0):
    accents_mapping = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u', 'ç': 'c',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U', 'Ñ': 'N',
        'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U', 'Ç': 'c',
    }

    def remove_accents(text):
        percentage = per * (maxProb - minProb)
        accent_positions = [m.start() for m in pattern.finditer(text)] # r'[áéíóúüàèìòùñçÁÉÍÓÚÜÑÀÈÌÒÙÇ]'
        num_to_replace = math.ceil(percentage * len(accent_positions))
        if num_to_replace == 0:
            return text  # Return the original text if no accents are to be removed

        remove_indices = set(random_state.choice(accent_positions, size=num_to_replace, replace=False))
        new_text = []

        for i in range(len(text)):
            if i in remove_indices:
                accented_char = text[i]
                unaccented_char = accents_mapping.get(accented_char, accented_char)  # Use mapped char or original if not found
                new_text.append(unaccented_char)
            else:
                new_text.append(text[i])

        return ''.join(new_text)

    return [remove_accents(text) for text in texts]

# Removes periods and commas (avoiding numbers), as well as ; : ' " - ...
def remove_punctuation(texts, random_state, per, pattern, minProb=0, maxProb=1.0):
    def replace_punctuation(text):
        percentage = per * (maxProb - minProb)
        punctuation_positions = [m.start() for m in pattern.finditer(text)] # r'[^\w\s]'
        num_to_replace = math.ceil(percentage * len(punctuation_positions))
        if num_to_replace == 0:
            return text

        replace_indices = set(random_state.choice(punctuation_positions, size=num_to_replace, replace=False))
        translation_table = {ord(text[pos]): None for pos in replace_indices}
        return text.translate(translation_table)

    return [replace_punctuation(text) for text in texts]


# Join sentences by removing periods (avoiding numbers) and ellipsis (...), as well as capital letters after them
def join_sentences(texts, random_state, per, pattern, minProb=0, maxProb=1.0):
    def join_sentences(text):
        percentage = per * (maxProb - minProb)
        # Focusing on periods and ... only for capital letter conversion
        punctuation_positions = [(m.start(), m.group()) for m in pattern.finditer(text)]  # r'(?<!\d)\.(?!\d)|(\.\.\.)', 
        num_to_replace = math.ceil(percentage * len(punctuation_positions))
        if num_to_replace == 0:
            return text
        remove_indices = set(random_state.choice([pos[0] for pos in punctuation_positions], size=num_to_replace, replace=False))
        
        new_text = []
        last_index = 0
        for pos, match in punctuation_positions:
            if pos in remove_indices:
                # Append the text up to the period
                new_text.append(text[last_index:pos])
                # Find the index of the next non-space character
                next_char_index = pos + (3 if match == '...' else 1)
                while next_char_index < len(text) and text[next_char_index] == ' ':
                    next_char_index += 1
                # Convert the next character to lowercase if it's a capital letter; ensure it doesn't exceed text length
                if next_char_index < len(text) and text[next_char_index].isupper():
                    new_text.append(' ' + text[next_char_index].lower())
                    last_index = next_char_index + 1  # Move past the modified character
                else:
                    last_index = pos + (3 if match == '...' else 1)  # No letter to change, proceed as usual
            else:
                new_text.append(text[last_index:pos + (3 if match == '...' else 1)])
                last_index = pos + (3 if match == '...' else 1)
        new_text.append(text[last_index:])  # Add the remainder of the text
        return ''.join(new_text)

    # Apply replacement to each text in the specified column of the batch
    return [join_sentences(text) for text in texts]

def lowercase(texts):
    return [text.lower() for text in texts]

# Simply removes capital letters
def remove_uppercase(texts, random_state, per, pattern, minProb=0, maxProb=1.0):
    def remove_uppercase(text):
        percentage = per * (maxProb - minProb)
        capital_positions = [(m.start(), m.group()) for m in pattern.finditer(text)] # r'[A-Z]',
        num_to_replace = math.ceil(percentage * len(capital_positions))

        if num_to_replace == 0:
            return text

        replace_indices = set(random_state.choice([pos[0] for pos in capital_positions], size=num_to_replace, replace=False))

        return ''.join([char.lower() if i in replace_indices else char for i, char in enumerate(text)])

    return [remove_uppercase(text) for text in texts]

# Removes accents, punctuation and capitalization
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

# Uses nltk.word_tokenize. Some datasets use this normalization and it'd be nice to adapt to it too
def normalize_space_tokenized(texts):
    return [" ".join(word_tokenize(text, language='spanish')) for text in texts]
