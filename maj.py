import os
import json
import random
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--target_n", type=int, default="8")
    parser.add_argument("--expand_factors", type=str, default="1,2,4,8")
    args = parser.parse_args()
    return args

def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set as {seed}")


def prepare_data(args):
    with open(args.input_file, 'r') as f:
        examples = json.load(f)
    return examples
    

def majority_voting(preds, scores):
    if len(preds) != len(scores):
        raise ValueError("The lists 'preds' and 'scores' must have the same length")
    
    # Filter out None predictions and gather the valid predictions and scores
    valid_entries = [(pred, score) for pred, score in zip(preds, scores) if pred is not None]
    if not valid_entries:
        return 0.0  # Return None if no valid predictions
    
    # Count occurrences of each prediction
    prediction_counts = {}
    prediction_scores = {}
    
    for pred, score in valid_entries:
        if pred not in prediction_counts:
            prediction_counts[pred] = 0
            prediction_scores[pred] = score
        prediction_counts[pred] += 1
    
    # Find the most common prediction
    max_count = 0
    majority_pred = None
    
    for pred, count in prediction_counts.items():
        if count > max_count:
            max_count = count
            majority_pred = pred
    
    return prediction_scores[majority_pred]


def pruning(sample_step_rewards, target_n, option="avg"):
    assert target_n <= len(sample_step_rewards)
    sample_rewards = [
        step_reward_aggregate(reward, option=option) for reward in sample_step_rewards
    ]
    indexed_sample_rewards = [(val, idx) for idx, val in enumerate(sample_rewards)]
    indexed_sample_rewards.sort(reverse=True)
    return [indexed_sample_rewards[i][1] for i in range(target_n)]


def main(args):
    samples = prepare_data(args)

    # All n_sampling
    avg_accs = []
    maj_accs = []
    for sample in samples:
        avg_accs.append(np.mean(sample["score"]))
        maj_accs.append(majority_voting(sample["pred"], sample["score"]))

    print(f"  Original || Acc: {np.mean(avg_accs):.4f} | Maj: {np.mean(maj_accs):.4f}\n")


    # Random pruning and expand
    expand_factors = [int(i) for i in args.expand_factors.split(",")]
    n_sampling = len(samples[0]["pred"])

    for expand_factor in expand_factors:
        target_n = args.target_n * expand_factor

        random_avg_accs = []
        random_maj_accs = []
        for sample in samples:
            aggregate_random_avg_accs = []
            aggregate_random_maj_accs = []
            for _ in range(100):
                pruned_inds = np.random.choice(n_sampling, target_n, replace=False)
                pruned_inds.sort()
                pruned_sample_preds = [sample["pred"][i] for i in pruned_inds]
                pruned_sample_scores = [sample["score"][i] for i in pruned_inds]

                aggregate_random_avg_accs.append(np.mean(pruned_sample_scores))
                aggregate_random_maj_accs.append(majority_voting(pruned_sample_preds, pruned_sample_scores))

            random_avg_accs.append(np.mean(aggregate_random_avg_accs))
            random_maj_accs.append(np.mean(aggregate_random_maj_accs))

        print(f"  Random n_sampling={target_n} || Acc: {np.mean(random_avg_accs):.4f} |  Maj: {np.mean(random_maj_accs):.4f}")

    print("\n")

    # Pruning and expand
    for expand_factor in expand_factors:
        target_n = args.target_n * expand_factor

        pruned_avg_accs = []
        pruned_maj_accs = []
        for sample in samples:
            aggregate_pruned_avg_accs = []
            aggregate_pruned_maj_accs = []
            for _ in range(100):
                pruned_inds = []
                for i in range(args.target_n):
                    pruned_inds += random.sample(range(i*args.target_n, (i+1)*args.target_n), expand_factor)
                pruned_inds.sort()
                pruned_sample_preds = [sample["prune_and_expand_pred"][i] for i in pruned_inds]
                pruned_sample_scores = [sample["prune_and_expand_scores"][i] for i in pruned_inds]

                aggregate_pruned_avg_accs.append(np.mean(pruned_sample_scores))
                aggregate_pruned_maj_accs.append(majority_voting(pruned_sample_preds, pruned_sample_scores))

            pruned_avg_accs.append(np.mean(aggregate_pruned_avg_accs))
            pruned_maj_accs.append(np.mean(aggregate_pruned_maj_accs))

        print(f"  Pruned n_sampling={target_n} || Acc: {np.mean(pruned_avg_accs):.4f} | Maj: {np.mean(pruned_maj_accs):.4f}")

        
if __name__ == "__main__":
    args = parse_args()
    set_seed(0)
    main(args)