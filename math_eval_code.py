import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
import datasets

from utils.data import construct_prompt
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math", type=str)
    parser.add_argument("--data_dir", default="./datas", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="deepseek-r1", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--min_p", default=0.05, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--disable_chunked_prefill', action='store_true', default=False)
    parser.add_argument('--max_model_len', type=int, default=40000)
    parser.add_argument("--n_sampling", type=int, default=1)

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
    dataset = datasets.load_dataset("json", data_files=["/mnt/nushare2/data/baliao/latb/lcb/test6.jsonl"], split="train")
    examples = []
    for p in dataset:
        new_p = {}
        for k, v in p.items():
            new_p[k] = str(v)
        examples.append(CodeGenerationProblem(**new_p))
    examples = sorted(examples, key=lambda x: x.question_id)

    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # get out_file name
    out_file_prefix = f"{args.prompt_type}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}_n{args.n_sampling}.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, out_file


def main(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if "magistral" in args.model_name_or_path.lower():
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
            gpu_memory_utilization=0.85,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )
    else:
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
            gpu_memory_utilization=0.85,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        evaluation(llm, data_name, args)


def get_question_template(question):
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```"
    else:
        prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```"
    return prompt


def evaluation(llm, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = i

        # parse question and answer
        question = get_question_template(example)
        full_prompt = construct_prompt({"question": question}, args)

        if i == 0:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": question,
            "prompt": full_prompt,
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

    # start inference
    prompts = [sample["prompt"] for sample in samples]
    start_time = time.time()

    if "magistral" in args.model_name_or_path.lower():
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens_per_call,
            n=args.n_sampling,
            seed=args.seed,
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens_per_call,
            n=args.n_sampling,
            skip_special_tokens=False,
            seed=args.seed,
        )

    outputs = llm.generate(
        prompts,
        sampling_params,
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    assert len(outputs) == len(prompts)
    end_time = time.time()

    results = []
    for sample, output in zip(samples, outputs):
        results.append(
            {
                "idx": sample["idx"],
                "question": sample["question"],
                "model_output": [o.text for o in output.outputs],
            }
        )

    time_use = (end_time - start_time) / 60
    print(f"Saving model outputs for {data_name} to {out_file}")
    json.dump(results, open(out_file, "w",), indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    main(args)