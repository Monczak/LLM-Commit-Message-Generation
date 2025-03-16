import json
import numpy as np
from numpy.linalg import norm
from collections import defaultdict
import time


vlan = 'vpy1no.jsonl'
vtrain = 'encoded_diffspy2.jsonl'
output_file = 'pybest_no_selectv.jsonl'
input_file = 'pytrain_no_selectv.jsonl'

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

def get_top_similar(test_vec, train_vectors, top_k=10):
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


# Assuming load_vectors and get_top_similar have been defined

# Load test set and training set vectors
test_vectors = load_vectors(vlan)
train_vectors = load_vectors(vtrain)

with open(output_file, 'a') as outfile:
    for test_id, test_vec in test_vectors.items():

        # Get the top 10 most similar training sets diff_id
        top_similar_ids = get_top_similar(test_vec, train_vectors, top_k=10)

        # Get the diff and msg of these diff_ids from pytrainyuan3.jsonl
        diff_msg_data = get_diff_msg(input_file, top_similar_ids)

        # Formatting results
        result = {"diff_id": test_id}
        for i, similar_id in enumerate(top_similar_ids, 1):
            result[f"best_id{i}"] = similar_id
            result[f"best_diff{i}"] = diff_msg_data[similar_id]['diff']
            result[f"best_msg{i}"] = diff_msg_data[similar_id]['msg']

        # Write single result to file
        outfile.write(json.dumps(result) + '\n')
