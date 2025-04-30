import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import transformers
from vllm import LLM, SamplingParams

from utils.data import load_data, construct_prompt
from utils.parser import parse_question, parse_ground_truth, extract_and_verify_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--data_dir", default="./datas", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--reward_model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="deepseek-r1", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--first_reasoning_end_idx", type=int, default=-1, help="-1 means using Alternatively.")
    
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args

def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")

def prepare_data(data_name, args):
    with open(args.input_file, 'r') as f:
        examples = json.load(f)
    
    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]

    # get out_file name
    out_file_prefix = args.input_file.split("/")[-1][:-len(".json")]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_reward{args.reward_model_name_or_path}_firstend{args.first_reasoning_end_idx}.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, out_file

def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    rm = LLM(
        model=args.reward_model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        task="reward",
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )
    rm_tokenizer = rm.get_tokenizer()

    # infer
    main(args, rm, rm_tokenizer, tokenizer)

def create_messages(query, response):
    """Create messages for the reward model."""
    response = response.strip()
    return [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": "<extra_0>".join(response.split("\n\n")) + "<extra_0>"},
    ]

def main(args, rm, rm_tokenizer, tokenizer):
    examples, out_file = prepare_data(args, args.data_name)
    print("=" * 50)
    print("data:", arg.data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        if example["question"] == "":
            continue

        if i == 0:
            print(example["question"])
      
        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt": example["gt"],
            "preds": example["pred"],
            "score": example["score"],
            "model_output": example["model_output"],
        }
        samples.append(sample)

    # Reward the think_summary, i.e. everything after </think>
    n_sampling = len(samples[0]["model_output"])
    for i, sample in enumerate(samples):
        sample_tok_messages = []
        error_ids = []
        for j, model_output in enumerate(sample["model_output"]):
            if "</think>" not in model_output:
                error_ids.append(j)
            else:
                response = model_output.split("</think>")[-1].strip()
                messages = create_messages(sample["question"], response)
                sample_tok_messages.append(
                    rm_tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                )

        if sample_tok_messages:
            rm_outputs = rm.encode(sample_tok_messages)
            sample_rewards = []
            count = 0
            for j in range(n_sampling):
                if j in error_ids:
                    sample_rewards.append([float("-inf")])
                else:
                    sample_rewards.append(list(rm_outputs[count].outputs.data[:, -1].numpy()))
                    count += 1
        else:
            sample_rewards = [[float("-inf")] for _ in range(n_sampling)]

        sample["reward"] = sample_rewards

  
    # Reward the first reasoning
    for i, sample in enumerate(samples):
        sample_tok_first_reasoning_messages = []
        for j, model_output in enumerate(sample["model_output"]):
            first_reasoning = tokenizer.decode(
                tokenizer.encode(model_output)[1:arg.first_reasoning_end_idx]
            )
            messages = create_messages(sample["question"], first_reasoning)            
            sample_tok_first_reasoning_messages.append(
                rm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            )

        rm_outputs = rm.encode(sample_tok_first_reasoning_messages)
        sample_rewards = []
        for j in range(n_sampling):
            sample_rewards.append(list(rm_outputs[j].outputs.data[:, -1].numpy()))

        sample["first_reasoning_reward"] = sample_rewards

  
    print(f"Saving rewards for {data_name} to {out_file}")
    json.dump(samples, open(out_file, "w",), indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)
