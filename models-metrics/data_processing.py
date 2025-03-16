from utils import process_diff, process_text

def process_item(item):
    """Process a single data item from the dataset."""
    diff_id = item['diff_id']
    diff = item['diff']
    cleaned_diff = process_diff(diff)
    
    # Process the commit message if available
    msg = item.get('msg', '')
    if msg:
        msg = process_text(msg)
    
    return {
        'diff_id': diff_id,
        'diff': cleaned_diff,
        'msg': msg
    }

def prepare_examples(best_examples, num_examples=5):
    """Prepare examples for few-shot learning."""
    examples = []
    
    for i in range(min(num_examples, len(best_examples))):
        best_diff = best_examples[i][0]
        best_msg = best_examples[i][1]
        examples.append({
            'role': 'user',
            'content': f"{best_diff}\nPlease write a commit message for the above code change."
        })
        examples.append({
            'role': 'assistant',
            'content': best_msg
        })
    
    return examples

def get_best_examples(diff_id, best_data):
    """Extract best examples for a given diff_id."""
    examples = []
    
    for item in best_data:
        if item['diff_id'] == diff_id:
            best_diff = process_diff(item["diff"])
            best_msg = process_text(item["msg"])
            examples.append((best_diff, best_msg))
    
    return examples
