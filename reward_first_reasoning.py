import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import transformers
from vllm import LLM



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--proxy_model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--first_reasoning_end_idx", type=int, default=512)
    parser.add_argument("--phrase", action="store_true", default=False)
    parser.add_argument("--is_orm", action="store_true", default=False)
    args = parser.parse_args()
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
        examples = examples[:args.num_test_sample]

    # get out_file name
    model_name = args.model_name_or_path.split("/")[-1]
    out_file_prefix = args.input_file.split("/")[-1][:-len(".json")]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{args.data_name}/{out_file_prefix}_rm{model_name}_firstend{args.first_reasoning_end_idx}_phrase{args.phrase}.json"
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
        task="reward",
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    proxy_tokenizer = transformers.AutoTokenizer.from_pretrained(args.proxy_model_name_or_path)

    # Reward
    main(args, llm, tokenizer, proxy_tokenizer)


def create_messages(query, response, is_orm=False):
    """Create messages for the reward model."""
    response = response.strip()
    if not is_orm:
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(response.split("\n\n")) + "<extra_0>"},
        ]
    else:
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]


def main(args, llm, tokenizer, proxy_tokenizer):
    examples, out_file = prepare_data(args)
    print("=" * 50)
    print("data:", args.data_name, " , #samples:", len(examples))

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
            "pred": example["pred"],
            "score": example["score"],
            #"reward": example["reward"],
            "model_output": example["model_output"],
        }
        samples.append(sample)

    # Reward the think_summary, i.e. everything after </think>
    n_sampling = len(samples[0]["model_output"])
    for i, sample in enumerate(samples):
        sample_tok_messages = []
        for j, model_output in enumerate(sample["model_output"]):
            if args.phrase:  # Using "Alternatively" as a definition for first step
                first_reasoning = model_output.split("Alternatively")[0]
                if len(proxy_tokenizer.encode(first_reasoning)) > 4096:  # Fallback to the first reasoning end index
                    first_reasoning = proxy_tokenizer.decode(
                        proxy_tokenizer.encode(model_output)[1:args.first_reasoning_end_idx]
                    ).strip()
            else:
                first_reasoning = proxy_tokenizer.decode(
                    proxy_tokenizer.encode(model_output)[1:args.first_reasoning_end_idx]
                ).strip()

            first_reasoning = "\n\n".join(first_reasoning.split("\n\n")[:-1])
            messages = create_messages(sample["question"], first_reasoning, is_orm=args.is_orm)
            sample_tok_messages.append(
                    tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
            )

        llm_outputs = llm.encode(sample_tok_messages)
        sample_rewards = []
        for j in range(n_sampling):
            if not args.is_orm:
                step_rewards = F.softmax(llm_outputs[j].outputs.data, dim=-1)[:, 1].tolist()
                sample_rewards.append([round(i, 5) for i in step_rewards])
            else:
                outcome_reward = llm_outputs[j].outputs.data[-1].numpy()
                sample_rewards.append([round(float(i), 5) for i in outcome_reward])

        sample["first_reasoning_reward"] = sample_rewards
  
    print(f"Saving first reasoning rewards for {args.data_name} to {out_file}")
    json.dump(samples, open(out_file, "w",), indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)