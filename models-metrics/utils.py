import re
import nltk
from nltk.tokenize import word_tokenize
import json

# Download required NLTK resources
nltk.download('punkt', quiet=True)

def is_camel_case(s):
    """Check if a string is in camelCase."""
    return s != s.lower() and s != s.upper() and "_" not in s

def to_underline(x):
    """Convert camelCase to space separated words."""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def process_text(text):
    """Normalize text by converting camelCase to spaced words."""
    words = text.split()
    processed_words = []
    
    for word in words:
        if len(word) > 1 and is_camel_case(word):
            processed_words.append(to_underline(word))
        else:
            processed_words.append(word)
            
    return ' '.join(processed_words)

def remove_between_identifiers(text, identifier_start, identifier_end):
    """Remove text between specific identifiers."""
    # Define regex patterns
    pattern = f'(?<={identifier_start}).*?(?={identifier_end})'

    # Use the re.sub method to replace the matched portion with the empty string
    result = re.sub(pattern, '', text)
    
    # Clean up residual markers
    if identifier_start == 'mmm a':
        result = result.replace('mmm a<nl>', '')
    if identifier_start == 'ppp b':
        result = result.replace('ppp b<nl>', '')
        result = result.replace('<nl>', '\n')
    
    # Clean up common formatting issues
    result = result.replace(' . ', '.')
    result = result.replace('  ', ' ')
    result = result.replace(' = ', '=')
    result = result.replace(' ; ', ';')
    result = result.replace(' (', '(')
    result = result.replace(') ', ')')
    
    return result

def get_tokens(text, max_tokens=1024):
    """Tokenize text and limit to max_tokens."""
    tokens = word_tokenize(text)
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    else:
        return ' '.join(tokens)

def process_diff(diff):
    """Process a code diff by removing identifiers and tokenizing."""
    # Normalize text
    diff = process_text(diff)
    
    # Remove identifiers
    result = remove_between_identifiers(diff, 'mmm a', '<nl>')
    cleaned_diff = remove_between_identifiers(result, 'ppp b', '<nl>')
    
    return get_tokens(cleaned_diff)

def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]

def write_jsonl(data, file_path, mode='w'):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def append_to_jsonl(data, file_path):
    """Append a single dictionary to a JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')
