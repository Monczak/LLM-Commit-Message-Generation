from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import json
import os

# Download required NLTK resources
nltk.download('punkt', quiet=True)

class CommitMessageEvaluator:
    """Evaluate commit messages using various metrics."""
    
    def __init__(self):
        self.rouge = Rouge()
    
    def calculate_meteor(self, reference, hypothesis):
        """
        Calculate a simplified METEOR score between two sentences.
        
        Args:
            reference (str): The reference message
            hypothesis (str): The generated message
            
        Returns:
            float: The METEOR score
        """
        if not reference or not hypothesis:
            return 0.0
            
        # Convert sentences into word frequency vectors
        vectorizer = CountVectorizer().fit([reference, hypothesis])
        reference_vector = vectorizer.transform([reference])
        hypothesis_vector = vectorizer.transform([hypothesis])
        
        # Compute cosine similarity
        similarity = cosine_similarity(reference_vector, hypothesis_vector)[0][0]
        
        # Calculate score using a simplified METEOR formula
        if len(reference) == 0 or len(hypothesis) == 0:
            return 0.0
            
        score = 2 * similarity * len(reference) * len(hypothesis) / (len(reference) + len(hypothesis))
        return score
    
    def calculate_bleu(self, reference, hypothesis):
        """
        Calculate BLEU score.
        
        Args:
            reference (str): The reference message
            hypothesis (str): The generated message
            
        Returns:
            float: The BLEU score
        """            
        return sentence_bleu([reference], hypothesis)
    
    def calculate_rouge_l(self, reference, hypothesis):
        """
        Calculate ROUGE-L score.
        
        Args:
            reference (str): The reference message
            hypothesis (str): The generated message
            
        Returns:
            dict: The ROUGE-L score with 'r', 'p', and 'f' values
        """
        scores = self.rouge.get_scores(hypothesis, reference, avg=True)
        return scores['rouge-l']
    
    def evaluate(self, reference, hypothesis):
        """
        Evaluate a generated commit message against a reference.
        
        Args:
            reference (str): The reference message
            hypothesis (str): The generated message
            
        Returns:
            dict: A dictionary containing the evaluation metrics
        """
        meteor = self.calculate_meteor(reference, hypothesis)
        bleu = self.calculate_bleu(reference, hypothesis)
        rouge_l = self.calculate_rouge_l(reference, hypothesis)
        
        return {
            'meteor': meteor,
            'bleu': bleu,
            'rouge_l': rouge_l
        }
    
    def calculate_average_metrics(self, eval_results):
        """
        Calculate average metrics across multiple evaluations.
        
        Args:
            eval_results (list): List of evaluation result dictionaries
            
        Returns:
            dict: Dictionary with average metric values
        """
        if not eval_results:
            return {
                'avg_meteor': 0.0,
                'avg_bleu': 0.0,
                'avg_rouge_l_f': 0.0,
                'avg_rouge_l_p': 0.0,
                'avg_rouge_l_r': 0.0,
                'count': 0
            }
        
        meteor_scores = [r['meteor'] for r in eval_results]
        bleu_scores = [r['bleu'] for r in eval_results]
        rouge_l_f_scores = [r['rouge_l']['f'] for r in eval_results]
        rouge_l_p_scores = [r['rouge_l']['p'] for r in eval_results]
        rouge_l_r_scores = [r['rouge_l']['r'] for r in eval_results]
        
        return {
            'avg_meteor': np.mean(meteor_scores),
            'avg_bleu': np.mean(bleu_scores),
            'avg_rouge_l_f': np.mean(rouge_l_f_scores),
            'avg_rouge_l_p': np.mean(rouge_l_p_scores),
            'avg_rouge_l_r': np.mean(rouge_l_r_scores),
            'count': len(eval_results)
        }

def evaluate_results(output_file, reference_field='msg', hypothesis_field='msgGPT0'):
    """
    Evaluate all results in a JSONL file.
    
    Args:
        output_file (str): Path to the JSONL file with results
        reference_field (str): Field name for reference messages
        hypothesis_field (str): Field name for generated messages
        
    Returns:
        dict: Dictionary with average metrics
    """
    evaluator = CommitMessageEvaluator()
    results = []
    
    # Check if file exists and has content
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        print(f"Warning: File {output_file} doesn't exist or is empty")
        return evaluator.calculate_average_metrics([])
    
    # Read and evaluate each entry
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Skip if either field is missing
            if reference_field not in data or hypothesis_field not in data:
                continue
                
            reference = data[reference_field]
            hypothesis = data[hypothesis_field]
            
            # Skip invalid entries
            if not reference or not hypothesis or reference == "0" or hypothesis == "0":
                continue
                
            eval_result = evaluator.evaluate(reference, hypothesis)
            results.append(eval_result)
    
    # Calculate and return average metrics
    avg_metrics = evaluator.calculate_average_metrics(results)
    
    # Print results
    print(f"\nEvaluation results for {output_file}:")
    print(f"  Number of evaluated examples: {avg_metrics['count']}")
    print(f"  Average METEOR score: {avg_metrics['avg_meteor']:.4f}")
    print(f"  Average BLEU score: {avg_metrics['avg_bleu']:.4f}")
    print(f"  Average ROUGE-L F1 score: {avg_metrics['avg_rouge_l_f']:.4f}")
    
    return avg_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate generated commit messages")
    parser.add_argument("--input", required=True, help="Path to the JSONL file with results")
    parser.add_argument("--ref-field", default="msg", help="Field name for reference messages")
    parser.add_argument("--hyp-field", default="msgGPT0", help="Field name for generated messages")
    
    args = parser.parse_args()
    
    evaluate_results(args.input, args.ref_field, args.hyp_field)
