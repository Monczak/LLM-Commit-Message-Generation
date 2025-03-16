from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import json
# Prepare your code diff text
from nltk.tokenize import word_tokenize
import json
import re
import nltk
import traceback
from openai import OpenAI
import time
import tqdm

lan = 'datasets/py.jsonl'
output_file = 'data/vpy1no.jsonl'

def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def to_Underline(x):
    """Rename by transposing spaces"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def remove_between_identifiers(text, identifier_start, identifier_end):
    # Define regex patterns
    pattern = f'(?<={identifier_start}).*?(?={identifier_end})'

    # Use the re.sub method to replace the matched portion with the empty string
    result = re.sub(pattern, '', text)
    if identifier_start == 'mmm a':
        result = result.replace('mmm a<nl>', '')
    if identifier_start == 'ppp b':
        result = result.replace('ppp b<nl>', '')
        result = result.replace('<nl>', '\n')
    result = result.replace(' . ', '.')
    result = result.replace('  ', '.')
    result = result.replace(' = ', '=')
    result = result.replace(' ; ', ';')
    result = result.replace(' (', '(')
    result = result.replace(') ', ')')
    return result
def get_tokens(text):
    if len(text) > 510:
        return ' '.join(text[:510])
    else:
        return ' '.join(text)
def process_diff(diff):
    wordsGPT = diff.split()
    msgGPT_list = []
    for wordGPT in wordsGPT:
        if len(wordGPT) > 1:
            if is_camel_case(wordGPT):
                msgGPT_list.append(to_Underline(wordGPT))
            else:
                msgGPT_list.append(wordGPT)
        else:
            msgGPT_list.append(wordGPT)
    diff = ' '.join(msgGPT_list)

    result = remove_between_identifiers(diff, 'mmm a', '<nl>')
    diff = remove_between_identifiers(result, 'ppp b', '<nl>')

    return get_tokens(diff)
def preprocess_code_diff(diff_text):
    """
    Preprocesses a code difference text by replacing lines starting with "+" with "[ADD]",
    lines starting with "-" with "[DEL]", and adding "[KEEP]" at the beginning of lines that
    do not start with "+" or "-".

    Parameters:
    diff_text (str): The original code difference text.

    Returns:
    str: The preprocessed code difference text.
    """
    # Split the text into individual lines
    lines = diff_text.split('\n')

    # Preprocess each line
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('+'):
            processed_lines.append('[ADD] ' + stripped_line[1:].strip())
        elif stripped_line.startswith('-'):
            processed_lines.append('[DEL] ' + stripped_line[1:].strip())
        else:
            processed_lines.append('[KEEP] ' + stripped_line)

    # Combine the processed lines back into a single string
    processed_diff = '\n'.join(processed_lines)

    return processed_diff


# Selection of appropriate pre-trained models and tokenizers
model_name = "microsoft/codereviewer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Read the JSON Lines file and vectorize each diff
with open(lan, 'r',encoding='utf8') as file, open(output_file, 'a') as outfile:
    for line in tqdm.tqdm(file):

        json_line = json.loads(line)

        if 'diff' in json_line and 'diff_id' in json_line:
            # Get diff field and diff_id
            diff = json_line['diff']
            result = remove_between_identifiers(diff, 'mmm a', '<nl>')
            code_diff1 = get_tokens(remove_between_identifiers(result, 'ppp b', '<nl>'))
            code_diff = preprocess_code_diff(code_diff1)
            diff_id = json_line['diff_id']

            # Tokenization of code difference text and moving it to the GPU
            input_ids = tokenizer.encode(code_diff, truncation=True, max_length=510, return_tensors="pt").to(device)

            # Get encoding vector
            model.eval()
            with torch.no_grad():
                encoder_outputs = model.encoder(input_ids)
            encoded_vector = encoder_outputs.last_hidden_state.cpu().numpy()
            # Get the first output as an encoded vector
            cls_vector = encoder_outputs.last_hidden_state[0, 0, :].cpu().numpy().tolist()

            # cls_vector is now a 1D array of 768 elements
            # print(len(cls_vector))
            # Create a dictionary with diff_id and encoding vector
            output_dict = {
                "diff_id": diff_id,
                "cls_vector": cls_vector
            }

            # Write dictionary to output file
            outfile.write(json.dumps(output_dict) + '\n')
