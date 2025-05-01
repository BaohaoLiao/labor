import json
import random
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--reward_option", choices=["avg", "min", "max", "last", "prod"], default="avg")
    parser.add_argument("--target_ns", type=str, default="1,2,4,8,16,32,64")
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
    bon_accs = []
    for sample in samples:
        sample_rewards = [
            step_reward_aggregate(reward, option=args.reward_option) for reward in sample["reward"]
        ]
        max_ind = sample_rewards.index(max(sample_rewards))

        avg_accs.append(np.mean(sample["score"]))
        bon_accs.append(sample["score"][max_ind])
        maj_accs.append(majority_voting(sample["pred"], sample["score"]))

    print(f"  Original || Acc: {np.mean(avg_accs):.4f} | BoN: {np.mean(bon_accs):.4f} | Maj: {np.mean(maj_accs):.4f}\n")


    # Random pruning
    target_ns = [int(i) for i in args.target_ns.split(",")]
    n_sampling = len(samples[0]["pred"])

    for target_n in target_ns:
        random_avg_accs = []
        random_maj_accs = []
        random_bon_accs = []
        for sample in samples:
            aggregate_random_avg_accs = []
            aggregate_random_maj_accs = []
            aggregate_random_bon_accs = []
            for _ in range(100):
                pruned_inds = random.sample(range(0, n_sampling), target_n)
                pruned_inds.sort()
                pruned_sample_preds = [sample["pred"][i] for i in pruned_inds]
                pruned_sample_scores = [sample["score"][i] for i in pruned_inds]
                pruned_sample_step_rewards = [sample["reward"][i] for i in pruned_inds]

                pruned_sample_rewards = [
                    step_reward_aggregate(reward, option=args.reward_option) for reward in pruned_sample_step_rewards
                ]
                max_ind = pruned_sample_rewards.index(max(pruned_sample_rewards))

                aggregate_random_avg_accs.append(np.mean(pruned_sample_scores))
                aggregate_random_bon_accs.append(pruned_sample_scores[max_ind])
                aggregate_random_maj_accs.append(majority_voting(pruned_sample_preds, pruned_sample_scores))

            random_avg_accs.append(np.mean(aggregate_random_avg_accs))
            random_bon_accs.append(np.mean(aggregate_random_bon_accs))
            random_maj_accs.append(np.mean(aggregate_random_maj_accs))

        print(f"  Random n_sampling={target_n} || Acc: {np.mean(random_avg_accs):.4f} | BoN: {np.mean(random_bon_accs):.4f} | Maj: {np.mean(random_maj_accs):.4f}")

    print("\n")

    # Pruning
    for target_n in target_ns:
        pruned_avg_accs = []
        pruned_maj_accs = []
        pruned_bon_accs = []
        for sample in samples:
            sample_first_reasoning_step_rewards = sample["first_reasoning_reward"]
            pruned_inds = pruning(sample_first_reasoning_step_rewards, target_n, option=args.reward_option)
            pruned_inds.sort()
            pruned_sample_preds = [sample["pred"][i] for i in pruned_inds]
            pruned_sample_scores = [sample["score"][i] for i in pruned_inds]
            pruned_sample_step_rewards = [sample["reward"][i] for i in pruned_inds]

            pruned_sample_rewards = [
                step_reward_aggregate(reward, option=args.reward_option) for reward in pruned_sample_step_rewards
            ]
            max_ind = pruned_sample_rewards.index(max(pruned_sample_rewards))

            pruned_avg_accs.append(np.mean(pruned_sample_scores))
            pruned_bon_accs.append(pruned_sample_scores[max_ind])
            pruned_maj_accs.append(majority_voting(pruned_sample_preds, pruned_sample_scores))

        print(f"  Pruned n_sampling={target_n} || Acc: {np.mean(pruned_avg_accs):.4f} | BoN: {np.mean(pruned_bon_accs):.4f} | Maj: {np.mean(pruned_maj_accs):.4f}")

        
if __name__ == "__main__":
    args = parse_args()
    main(args)