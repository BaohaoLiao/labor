import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

from utils.data import load_data, construct_prompt
from utils.parser import parse_question, parse_ground_truth, extract_and_verify_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--data_dir", default="./datas", type=str)
    parser.add_argument("--input_file", type=str, default=None)
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
    parser.add_argument('--max_model_len', type=int, default=64000)
    
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
    if "math500_level" in args.data_name:
        level = int(args.data_name.strip()[-1])
        examples = load_data("math500", args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(args.data_name, args.data_dir)

    with open(args.input_file, 'r') as f:
        baseline_results = json.load(f)
    
    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]
        baseline_results = baseline_results[:args.num_test_sample]

    # get out_file name
    out_file_prefix = args.input_file.split("/")[-1][:-len(".json")]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{args.data_name}/{out_file_prefix}_first_reasoning.json"
    os.makedirs(f"{output_dir}/{args.data_name}", exist_ok=True)
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
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # infer
    main(args, llm, tokenizer)


def main(args, llm, tokenizer):
    examples, baseline_results, out_file = prepare_data(args)
    print("=" * 50)
    print("data:", args.data_name, " , #samples:", len(examples))

    samples = []
    for i, (example, baseline_result) in tqdm(enumerate(zip(examples, baseline_results)), total=len(examples)):
        # parse question and answer
        example["question"] = parse_question(example, args.data_name)
        if example["question"] == "":
            continue
        full_prompt = construct_prompt(example, args)

        if i == 0:
            print(full_prompt)

        preds = []
        scores = []
        model_outputs = []
        for i, model_output in enumerate(baseline_result["model_output"]):
            if "Alternatively" in model_output:
                preds.append(baseline_result["preds"][i])
                scores.append(baseline_result["score"][i])
                model_outputs.append(model_output)

        assert len(preds) > 0
      
        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt": example["answer"],
            "prompt": full_prompt,
            "pred": preds,
            "score": score,
            "model_output": model_outputs,
        }
        samples.append(sample)

    # Sample answer for first reasoning
    prompt_and_first_reasonings = []
    n_samplings = [0]
    for i, sample in enumerate(samples):
        n_samplings.append(n_samplings[-1] + len(sample["model_output"]))
        for j, model_output in enumerate(sample["model_output"]):
            first_reasoning = model_output.split("Alternatively")[0].rstrip()
            prompt_and_first_reasonings.append(sample["prompt"] + first_reasoning + "\n\n</think>")

    print(f"Num samplings for different questions: {n_samplings}")

    # start inference
    start_time = time.time()
    outputs = llm.generate(
        prompt_and_first_reasonings,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            skip_special_tokens=False,
            seed=args.seed,
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    assert len(outputs) == len(prompt_and_first_reasonings)
    end_time = time.time()

    first_reasoning_think_summarys = []
    for i in range(len(samples)):
        first_reasoning_think_summarys.append(
            [output.outputs[0].text for output in outputs[n_samplings[i] : n_samplings[i+1]]]
        )    

    # Extract pred and eval
    results = []
    avg_acc = []
    for sample, output in zip(samples, first_reasoning_think_summarys):
        gt = parse_ground_truth(sample, args.data_name)

        preds = []
        scores = []
        for o in output:
            pred, score = extract_and_verify_pred(o, gt, args.data_name)
            preds.append(pred)
            scores.append(score)
        avg_acc.append(np.mean(scores))

        results.append(
            {
                "idx": sample["idx"],
                "question": sample["question"],
                "gt": str(sample["answer"]),
                "first_reasoning_pred": preds,
                "final_pred": sample["pred"],
                "first_reasoning_score": scores,
                "final_score": sample["score"],
                "first_reasoning_summary": output,
                "conventional_thinking_and_summary": sample["model_output"],
            }
        )

    time_use = (end_time - start_time) / 60
    result_json = {
        "num_samples": len(samples),
        "pass@1": np.mean(avg_acc),
        "time_use_in_min": time_use,
    }
    print(result_json)

    print(f"Saving first reasoning for {args.data_name} to {out_file}")
    json.dump(results, open(out_file, "w",), indent=4)

    with open(out_file.replace(".json", "_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)
