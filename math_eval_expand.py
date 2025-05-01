import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

from utils.data import construct_prompt
from utils.parser import parse_ground_truth, extract_and_verify_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--data_dir", default="./datas", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="deepseek-r1", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--min_p", default=0.05, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--disable_chunked_prefill', action='store_true', default=False)
    parser.add_argument('--max_model_len', type=int, default=64000)

    # For good first reasoning
    parser.add_argument("--rewards_operation", choices=["min", "avg", "prod", "last"], default="avg")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--first_reasoning_end_idx", type=int, default=512)
    parser.add_argument("--target_n", type=int, default=1)
    parser.add_argument("--expand_factor", type=int, default=1)

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


def prepare_data(args):
    with open(args.input_file, 'r') as f:
        examples = json.load(f)

    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # get out_file name
    out_file_prefix = args.model_name_or_path.split("/")[-1]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{args.data_name}/{out_file_prefix}_targetn{args.target_n}_expand{args.expand_factor}.json"
    os.makedirs(f"{output_dir}/{args.data_name}", exist_ok=True)
    return examples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # Infer 
    main(args, llm, tokenizer)


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


def pruning(sample_step_rewards, target_n, option="avg"):
    assert target_n <= len(sample_step_rewards)
    sample_rewards = [
        step_reward_aggregate(reward, option=option) for reward in sample_step_rewards
    ]
    indexed_sample_rewards = [(val, idx) for idx, val in enumerate(sample_rewards)]
    indexed_sample_rewards.sort(reverse=True)
    return [indexed_sample_rewards[i][1] for i in range(target_n)]


def main(args, llm, tokenizer):
    examples, out_file = prepare_data(args)
    print("=" * 50)
    print("data:", args.data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = int(example["idx"])

        # parse question and answer
        if example["question"] == "":
            continue
        full_prompt = construct_prompt(example, args)

        if i == 0:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": example["gt"],
            "anwer": example["gt"],
            "prompt": full_prompt,
            "pred": example["pred"],
            "score": example["score"],
            "model_output": example["model_output"],
            "first_reasoning_reward": example["first_reasoning_reward"],
        }
        samples.append(sample)

    # Pruning
    prompts = []
    for sample in samples:
        pruned_ids = pruning(
            sample["first_reasoning_reward"], 
            args.target_n, 
            option=args.rewards_operation
        )
        pruned_ids.sort()
        for id in pruned_ids:
            first_reasoning = tokenizer.decode(
                tokenizer.encode(sample["model_output"][id])[1:args.first_reasoning_end_idx]
            )
            prompts.append(sample["prompt"] + first_reasoning)

    # start inference
    start_time = time.time()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_tokens=args.max_tokens_per_call,
            n=args.expand_factor,
            skip_special_tokens=False,
            seed=args.seed,
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    assert len(outputs) == len(prompts)
    end_time = time.time()

    # Reorder
    model_outputs = []
    for i in range(len(samples)):
        sample_model_outputs = []
        for j in range(args.target_n):
            sample_model_outputs += [o.text for o in outputs[i*args.target_n + j].outputs]
        model_outputs.append(sample_model_outputs)

    # Extract pred and eval
    results = []
    avg_acc = []
    for sample, sample_model_outputs in zip(samples, model_outputs):
        gt = parse_ground_truth(sample, args.data_name)

        preds = []
        scores = []
        for model_output in sample_model_outputs:
            # Avoid the bug in math_verify for multiple boxeds
            if "</think>" in model_output:
                model_output = model_output.split("</think>")[-1]
            pred, score = extract_and_verify_pred(model_output, gt, args.data_name)
            preds.append(pred)
            scores.append(score)
        avg_acc.append(np.mean(scores))

        results.append(
            {
                "idx": sample["idx"],
                "question": sample["question"],
                "gt": str(sample["answer"]),
                "pred": sample["preds"],
                "score": sample["scores"],
                "prune_and_expand_pred": preds,
                "prune_and_expand_scores": scores,
                "model_output": sample["model_output"],
                "prune_and_expand_model_output": sample_model_outputs,
            }
        )

    time_use = (end_time - start_time) / 60
    result_json = {
        "num_samples": len(samples),
        "pass@1": np.mean(avg_acc),
        "time_use_in_min": time_use,
    }
    print(result_json)

    print(f"Saving model outputs for {args.data_name} to {out_file}")
    json.dump(results, open(out_file, "w",), indent=4)

    with open(out_file.replace(".json", f"_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)