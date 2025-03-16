import json
import os
import re
import time
import traceback
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize

# Initialize NLTK components
nltk.download('punkt')

key = ''
lan = 'py.jsonl'
output_filename = 'pyaddresult_gemini.jsonl'

def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def to_Underline(x):
    """Rename by transposing spaces"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def get_tokens(text):
    tokens = word_tokenize(text)
    if len(tokens) > 1024:
        return ' '.join(tokens[:1024])
    else:
        return ' '.join(tokens)

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

# Setting up the Gemini API
genai.configure(key)
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_output_tokens": 50
}
model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)

# Read and process code difference data
with open(lan, 'r') as f:
    json_data = f.readlines()

for item in json_data:
    data = json.loads(item)
    diff_id = data['diff_id']
    diff = data['diff']
    # Apply preprocessing steps
    result = remove_between_identifiers(diff, 'mmm a', '<nl>')
    diff = get_tokens(remove_between_identifiers(result, 'ppp b', '<nl>'))

    if len(best_diffs_msgs) >= num_examples:
        # Building the Prompt
        prompt = ""
        for best_diff, best_msg in best_diffs_msgs[:num_examples]:
            prompt += f"{best_diff}\nPlease write a commit message for the above code change.\n{best_msg}\n\n"
        prompt += f"{diff}\nPlease write a commit message for the above code change.\n"

        generated_msg = None
        attempt = 0
        while attempt < 10 and generated_msg is None:
            try:
                response = model.generate_content([prompt])
                generated_msg = response.text.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} Failed: {e}")
                attempt += 1
                time.sleep(1)  # Simple wait mechanism to avoid retrying too quickly

        # If the message is successfully generated or the maximum number of attempts is reached, save the result to the appropriate file
        if generated_msg is not None:
            output_data = {
                "diff_id": diff_id,
                f"generated_msg_{num_examples}": generated_msg
            }
            with open(output_filename[num_examples], 'a', encoding='utf8') as f:
                json.dump(output_data, f)
                f.write('\n')
        else:
            print(f"Could not generate a message for diff_id {diff_id}.")
    time.sleep(1)
