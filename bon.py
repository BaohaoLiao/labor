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


def main(args):
    samples = prepare_data(args)

    avg_accs = []
    for sample in samples:
        sample_rewards = [
            step_reward_aggregate(reward, option=args.reward_option) for reward in sample["reward"]
        ]
        max_ind = sample_rewards.index(max(sample_rewards))
        avg_accs.append(sample["score"][max_ind])

    print(f"BoN is {np.mean(avg_accs):.4f}")

        
if __name__ == "__main__":
    args = parse_args()
    main(args)