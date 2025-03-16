import argparse
import logging
import sys
import time
import json
from pathlib import Path

from utils import read_jsonl, write_jsonl, append_to_jsonl
from data_processing import process_item, get_best_examples, prepare_examples
from model_clients import get_client
from metrics import CommitMessageEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate commit messages using LLMs')
    
    # Input/output options
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file with code diffs')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file for generated messages')
    parser.add_argument('--examples', type=str, help='JSONL file with example diffs and messages')
    
    # Model options
    parser.add_argument('--client', type=str, default='ollama', choices=['openai', 'mock'], 
                        help='Client to use for inference')
    parser.add_argument('--model', type=str, default='llama2', help='Model to use for inference')
    parser.add_argument('--base-url', type=str, default='http://localhost:11434/v1', 
                        help='Base URL for the API service')
    
    # Generation options
    parser.add_argument('--num-examples', type=int, default=5, 
                        help='Number of examples to use for few-shot learning (0 for zero-shot)')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Temperature for text generation')
    parser.add_argument('--max-tokens', type=int, default=50, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--top-p', type=float, default=0.95, 
                        help='Top-p sampling parameter')
    parser.add_argument('--num-generations', type=int, default=5, 
                        help='Number of messages to generate per diff')
    
    # Execution options
    parser.add_argument('--eval', action='store_true', help='Evaluate generated messages')
    parser.add_argument('--retry', type=int, default=3, help='Number of retries for API calls')
    
    return parser.parse_args()

def generate_messages(args):
    """Generate commit messages using the specified model."""
    # Initialize the client
    client = get_client(args.client, model=args.model, base_url=args.base_url)
    
    # Load data
    data = read_jsonl(args.input)
    logger.info(f"Loaded {len(data)} items from {args.input}")
    
    # Load examples if provided
    examples_data = []
    if args.examples and args.num_examples > 0:
        examples_data = read_jsonl(args.examples)
        logger.info(f"Loaded {len(examples_data)} examples from {args.examples}")
    
    # Initialize evaluator if evaluation is requested
    evaluator = CommitMessageEvaluator() if args.eval else None
    all_eval_results = []
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Clear output file if it exists
    with open(args.output, 'w') as f:
        pass
    
    # Process each item
    for i, item in enumerate(data):
        processed_item = process_item(item)
        diff_id = processed_item['diff_id']
        diff = processed_item['diff']
        reference_msg = processed_item.get('msg', '')
        
        logger.info(f"Processing item {i+1}/{len(data)}, diff_id: {diff_id}")
        
        # Get examples for this diff if available
        best_examples = []
        if args.num_examples > 0 and examples_data:
            best_examples = get_best_examples(diff_id, examples_data)
            if not best_examples:
                logger.warning(f"No examples found for diff_id {diff_id}")
        
        # Prepare messages for the model
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a programmer who makes the above code changes."
        })
        
        # Add examples if available
        if best_examples and args.num_examples > 0:
            messages.extend(prepare_examples(best_examples, args.num_examples))
        
        # Add the current diff
        messages.append({
            "role": "user",
            "content": f"{diff}\nPlease write a commit message for the above code change. Your response should only include the commit message and nothing else."
        })
        
        # Generate messages
        success = False
        retries = 0
        while not success and retries <= args.retry:
            try:
                generated_messages = client.generate(
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=args.num_generations
                )
                success = True
            except Exception as e:
                retries += 1
                logger.error(f"Error generating messages (attempt {retries}/{args.retry}): {e}")
                time.sleep(2 ** retries)  # Exponential backoff
        
        if not success:
            logger.error(f"Failed to generate messages for diff_id {diff_id} after {args.retry} attempts")
            generated_messages = [""] * args.num_generations
        
        # Evaluate if requested
        eval_results = []
        if args.eval and reference_msg:
            for msg in generated_messages:
                if msg:
                    result = evaluator.evaluate(reference_msg, msg)
                    eval_results.append(result)
            
            if eval_results:
                all_eval_results.extend(eval_results)
        
        # Save results
        output_data = {
            "diff_id": diff_id,
            "reference_msg": reference_msg,
        }
        
        for i, msg in enumerate(generated_messages):
            output_data[f"generated_msg_{i}"] = msg
        
        if eval_results:
            avg_metrics = evaluator.calculate_average_metrics(eval_results)
            for k, v in avg_metrics.items():
                if k != 'count':
                    output_data[k] = v
        
        append_to_jsonl(output_data, args.output)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Print summary of evaluation if requested
    if args.eval and all_eval_results:
        avg_metrics = evaluator.calculate_average_metrics(all_eval_results)
        logger.info("Evaluation summary:")
        logger.info(f"  Average METEOR score: {avg_metrics['avg_meteor']:.4f}")
        logger.info(f"  Average BLEU score: {avg_metrics['avg_bleu']:.4f}")
        logger.info(f"  Average ROUGE-L F1 score: {avg_metrics['avg_rouge_l_f']:.4f}")
        logger.info(f"  Number of evaluations: {avg_metrics['count']}")

def main():
    """Main entry point."""
    args = parse_args()
    try:
        generate_messages(args)
    except Exception as e:
        logger.error(f"Error running generate_messages: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()