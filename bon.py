import json
import argparse
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--reward_option", choices=["avg", "min", "max", "last", "prod"], default="avg")
    args = parser.parse_args()
    return args


def prepare_data(args):
    with open(args.input_file, 'r') as f:
        examples = json.load(f)
    return examples


def step_reward_aggregate(step_reward, option="avg"):
    if option == "avg":
        return np.mean(step_reward)
    elif option == "min":
        return np.min(step_reward)
    elif option == "max":
        return np.max(step_reward)
    elif option == "last":
        return step_reward[-1]
    else:
        return np.prod(step_reward)
    

def majority_voting(preds, scores):
    if len(preds) != len(scores):
        raise ValueError("The lists 'preds' and 'scores' must have the same length")
    
    # Filter out None predictions and gather the valid predictions and scores
    valid_entries = [(pred, score) for pred, score in zip(preds, scores) if pred is not None]
    if not valid_entries:
        return None, 0.0  # Return None if no valid predictions
    
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
    
    return prediction_counts[majority_pred]


def main(args):
    samples = prepare_data(args)

    avg_accs = []
    maj_accs = []
    bon_accs = []
    for sample in samples:
        avg_accs.append(np.mean(sample["score"]))

        sample_rewards = [
            step_reward_aggregate(reward, option=args.reward_option) for reward in sample["reward"]
        ]
        max_ind = sample_rewards.index(max(sample_rewards))
        bon_accs.append(sample["score"][max_ind])

        maj_accs.append(majority_voting(sample["pred"], sample["score"]))

    print(f"Acc: {np.mean(avg_accs):.4f} | BoN: {np.mean(bon_accs):.4f} | Maj: {np.mean(maj_accs):.4f}")

        
if __name__ == "__main__":
    args = parse_args()
    main(args)