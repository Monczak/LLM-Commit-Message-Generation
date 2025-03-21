from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import re
import numpy as np
from numpy.linalg import norm
import nltk
from openai import OpenAI
import time
import traceback

OPENAI_BASE_URL = "http://localhost:11434/v1" # For use with Ollama
LLM_MODEL = "llama3.2:3b"

sitekey = 'sduoj'

# Selection of appropriate pre-trained models and tokenizers
model_name = "microsoft/codereviewer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def to_Underline(x):
    """转空格命名"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def remove_between_identifiers(text, identifier_start, identifier_end):
    # Define regex patterns
    pattern = f'(?<={identifier_start}).*?(?={identifier_end})'
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
    tokens = nltk.word_tokenize(text)
    if len(tokens) > 600:
        return ' '.join(tokens[:600])
    else:
        return ' '.join(tokens)

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

def generate_vector(code_text):
    # Preprocessing code difference text
    result = remove_between_identifiers(code_text, 'mmm a', '<nl>')
    code_diff1 = get_tokens(remove_between_identifiers(result, 'ppp b', '<nl>'))
    code_diff = preprocess_code_diff(code_diff1)

    # Tokenization of code text and moving it to the GPU
    input_ids = tokenizer.encode(code_diff, truncation=True, max_length=510, return_tensors="pt").to(device)

    # Get encoding vector
    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids)
    cls_vector = encoder_outputs.last_hidden_state[0, 0, :].cpu().numpy().tolist()

    return cls_vector

def load_vectors(jsonl_file):
    """Load diff_id and vector from JSONL file"""
    vectors = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            vectors[data['diff_id']] = np.array(data['cls_vector'])
    return vectors

def cosine_similarity(vec_a, vec_b):
    """Compute the cosine similarity of two vectors"""
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def get_top_similar(test_vec, train_vectors, top_k=5):
    """Get the diff_id of the top_k training vectors that are most similar to the test vector"""
    similarities = {train_id: cosine_similarity(test_vec, train_vec)
                    for train_id, train_vec in train_vectors.items()}
    # Sort by similarity and take the first top_k
    return sorted(similarities, key=similarities.get, reverse=True)[:top_k]

def get_diff_msg(jsonl_file, diff_ids):
    """Get diff and msg for a specific diff_id from a JSONL file"""
    diff_msg = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['diff_id'] in diff_ids:
                diff_msg[data['diff_id']] = {'diff': data['diff'], 'msg': data['msg']}
    return diff_msg


# review.py

import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)
progress = 0


def save_to_json(site_key, code_diff, commit_message):
    record = {
        "site_key": site_key,
        "code_diff": code_diff,
        "commit_message": commit_message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open('commit_messages.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')


def generate_commit_message(code_text, diff_id, train_vectors_file, diff_msg_file, api_key, site_key):
    global progress
    if site_key != sitekey:
        return "Invalid site key"

    progress = 10
    test_vector = generate_vector(code_text)
    train_vectors = load_vectors(train_vectors_file)
    progress = 30
    top_similar_ids = get_top_similar(test_vector, train_vectors, top_k=5)
    diff_msg_data = get_diff_msg(diff_msg_file, top_similar_ids)
    progress = 50

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=api_key,
    )

    messages = [
        {"role": "system", "content": "You are a programmer who makes the above code changes."}
    ]

    for diff_id in top_similar_ids:
        best_diff = diff_msg_data[diff_id]['diff']
        best_msg = diff_msg_data[diff_id]['msg']
        messages.append({"role": "user",
                         "content": f"{best_diff}\nPlease write a commit message that contains only one simple sentence for the above code change."})
        messages.append({"role": "assistant", "content":f"{best_msg}"})

    messages.append({"role": "user",
                     "content": f"{code_text}\nPlease write a commit message that contains only one simple sentence for the above code change.\n"})

    print(messages)

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=128,
        temperature=0.8,
        n=1,
        top_p=0.95
    )

    final_commit_message = completion.choices[0].message.content.strip()
    progress = 90

    save_to_json(site_key, code_text, final_commit_message)
    progress = 100

    return final_commit_message


@app.route('/progress')
def get_progress():
    global progress
    return jsonify({"progress": progress})


if __name__ == "__main__":
    # For testing purpose
    code_text = """
    public boolean indexMetaDataChanged ( IndexMetaData current ) { <nl> if ( previousIndexMetaData = = current ) { <nl> return false ; <nl> } <nl> - return false ; <nl> + return true ; <nl> } <nl> <nl> public boolean blocksChanged ( ) { <nl>
    """
    diff_id = "java"
    train_vectors_file = 'vtrain_java.jsonl'
    diff_msg_file = 'javatrainyuan3.jsonl'
    api_key = "your-openai-api-key"
    site_key = "sduoj"
    print(generate_commit_message(code_text, diff_id, train_vectors_file, diff_msg_file, api_key, site_key))
