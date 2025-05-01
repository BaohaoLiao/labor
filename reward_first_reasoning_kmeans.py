import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import transformers



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--first_reasoning_end_idx", type=int, default=512)
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
    out_file = f"{output_dir}/{args.data_name}/{out_file_prefix}_embed{model_name}_firstend{args.first_reasoning_end_idx}.json"
    os.makedirs(f"{output_dir}/{args.data_name}", exist_ok=True)
    return examples, out_file


def setup(args):
    # load model
    llm = transformers.AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map='balanced',
        torch_dtype=torch.half,
    ).eval().requires_grad_(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    main(args, llm, tokenizer)


@torch.no_grad()
def obtain_embeddings(llm, tokenizer, responses):
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    batch_dict = tokenizer(
        ["query: " + r for r in responses], 
        max_length=512, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = llm(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def main(args, llm, tokenizer):
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
            "reward": example["reward"],
            "model_output": example["model_output"],
            "first_reasoning_reward": example["first_reasoning_reward"],
        }
        samples.append(sample)

    # Reward the think_summary, i.e. everything after </think>
    n_sampling = len(samples[0]["model_output"])
    for i, sample in enumerate(samples):
        embeddings = obtain_embeddings(llm, tokenizer, sample["model_output"])
        sample["emb"] = embeddings
  
    print(f"Saving first reasoning embedding for {args.data_name} to {out_file}")
    json.dump(samples, open(out_file, "w",), indent=4)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)