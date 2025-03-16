import json
import os
import re
import time
import traceback
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize

lan = 'py.jsonl'
output_filenames = {
    1: 'pyaddresult_gemini_1nov.jsonl',
    3: 'pyaddresult_gemini_3nov.jsonl',
    5: 'pyaddresult_gemini_5nov.jsonl',
    10: 'pyaddresult_gemini_10nov.jsonl'
}
key = ' '
best_file = 'pybest_no_selectv.jsonl'
# Initialize NLTK components
nltk.download('punkt')

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

    # Extract the corresponding best_diff and best_msg
    best_diffs_msgs = []
    with open(best_file, 'r') as file:
        for line in file:
            best_data = json.loads(line)
            if best_data['diff_id'] == diff_id:
                for i in range(1, 11):
                    diff_key = f'best_diff{i}'
                    msg_key = f'best_msg{i}'
                    if diff_key in best_data and msg_key in best_data:
                        # Apply the same pre-processing steps
                        result_b = remove_between_identifiers(best_data[diff_key], 'mmm a', '<nl>')
                        best_diff = get_tokens(remove_between_identifiers(result_b, 'ppp b', '<nl>'))
                        best_msg = best_data[msg_key]
                        best_diffs_msgs.append((best_diff, best_msg))
                break

    # Generate commit messages based on a varying number of examples
    for num_examples in [1,3,5,10]:
        if len(best_diffs_msgs) >= num_examples:
            # Build the prompt
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
                with open(output_filenames[num_examples], 'a', encoding='utf8') as f:
                    json.dump(output_data, f)
                    f.write('\n')
            else:
                print(f"Could not generate a message for diff_id {diff_id}.")
        time.sleep(1)
