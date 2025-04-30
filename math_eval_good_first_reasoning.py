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
    parser.add_argument("--n_sampling", type=int, default=1)

    # For good first reasoning
    parser.add_argument("--reward_model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--rewards_operation", choices=["min", "avg", "prod", "last"], default="avg")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--start_token_idx", type=int, default=-1, help="-1 means using Alternatively.")
    parser.add_argument("--n_retain", type=int, default=1)
    parser.add_argument("--expand_factor", type=int, default=1)
    
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    assert args.n_retain < args.n_sampling and args.n_retain > 0
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
    # Load benchmark set
    if "math500_level" in data_name:
        level = int(data_name.strip()[-1])
        examples = load_data("math500", args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(data_name, args.data_dir)

    # Load baseline results
    with open(args.input_file, 'r') as f:
        baseline_results = json.load(f)
    
    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]
        baseline_results = baseline_results[:args.num_test_sample]

    # get out_file name
    out_file_prefix = f"{args.prompt_type}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}_n{args.n_sampling}_start{args.start_token_idx}_nretrain{args.n_retain}_expand{args.expand_factor}.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, baseline_results, out_file


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

    # Reward model
    rm_tokenizer = transformers.AutoTokenizer.from_pretrained(args.reward_model_name_or_path, trust_remote_code=True)
    rm = transformers.AutoModel.from_pretrained(
        args.reward_model_name_or_path, 
        device_map="cuda:0", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    # infer & eval
    main(args, llm, tokenizer, rm, rm_tokenizer)


@torch.no_grad()
def select_better_first_reasoning(rm, rm_tokenizer, question, reasonings, reward_operation, n_retain):
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
  
    rewards = []
    for reasoning in reasonings:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(reasoning.strip().split("\n\n")) + "<extra_0>"},
        ]
        conversation_str = rm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        input_ids = tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        ).to(rm.device)
        outputs = rm(input_ids=input_ids)
        step_sep_id = rm_tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        step_reward = make_step_rewards(outputs[0], token_masks)[0]

        if args.reward_operation == "avg":
            reward = np.mean(step_reward)
        elif args.reward_operation == "last":
            reward = step_reward[-1]
        elif args.reward_operation == "min":
            reward = np.min(step_reward)
        else
            reward = np.prod(step_reward)
        rewards.append(reward)

    paired = list(zip(rewards, reasonings))
    paired.sort(reverse=True)
    selected_reasonings = [reasoning for _, reasoning in paired[:n_retain]]

    return rewards, selected_reasonings


def main(args, llm, tokenizer, rm, rm_tokenizer):
    examples, baseline_results, out_file = prepare_data(args, args.data_name)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for i, (example, baseline_result) in tqdm(enumerate(zip(examples, baseline_results)), total=len(examples)):
        idx = int(example["idx"])

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        full_prompt = construct_prompt(example, args)

        if i == 0:
            print(full_prompt)

        # TODO: check
        if arg.start_token_idx != -1:
            first_reasonings = [
                tokenizer.decode(
                    tokenizer.encode(o)[1:arg.start_token_idx]
                ) 
            ] for o in baseline_result["model_output"]]

        # Select better first reasonings
        rewards, selected_first_reasonings = select_better_first_reasoning(
              rm, 
              rm_tokenizer, 
              example["question"], 
              first_reasonings, 
              args.reward_operation, 
              args.n_retain
        )
      
        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": example["answer"],
            "prompt": full_prompt,
            "first_reasoning_reward": rewards,
            "first_reasoning": first_reasonings,
            "selected_first_reasoning": selected_first_reasonings,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)
    
    prompts = []
    for sample in samples:
        prompts += [sample["prompt"] + sample["selected_first_reasoning"][i] for i in range(args.n_retain)]

    # start inference
    start_time = time.time()
    llm_outputs = llm.generate(
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
    llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
    assert len(llm_outputs) == len(prompts)
    end_time = time.time()

    # Reorder
    model_outputs = []
    for i in range(len(samples)):
        sample_model_outputs = []
        for j in range(args.n_retain):
            sample_model_outputs += [o.text for o in llm_outputs[i*args.n_retain + j].outputs]
        model_outputs.append(sample_model_outputs)
        
    # Extract pred and eval
    results = []
    avg_acc = []
    for sample, sample_model_outputs in zip(samples, model_outputs):
        gt = parse_ground_truth(sample, data_name)

        preds = []
        scores = []
        for o in sample_model_outputs:
            # Avoid the bug in math_verify for multiple boxeds
            if "</think>" in o:
                model_output = o.split("<\think>")[-1]
            else:
                model_output = o
            pred, score = extract_and_verify_pred(model_output, gt, data_name)
            preds.append(pred)
            scores.append(score)
        avg_acc.append(np.mean(scores))

        results.append(
            {
                "idx": sample["idx"],
                "question": sample["question"],
                "gt": str(sample["answer"]),
                "preds": preds,
                "score": scores,
                "first_reasoning_reward": rewards,
                "first_reasoning": first_reasonings,
                "selected_first_reasoning": selected_first_reasonings,
                "model_output": sample_model_outputs,
            }
        )

    time_use = (end_time - start_time) / 60
    result_json = {
        "num_samples": len(samples),
        "pass@1": np.mean(avg_acc),
        "time_use_in_min": time_use,
    }
    print(result_json)

    print(f"Saving model outputs for {data_name} to {out_file}")
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
