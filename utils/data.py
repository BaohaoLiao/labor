import os
import json
from pathlib import Path
from typing import Iterable, Union, Any


PROMPT_TEMPLATES = {
    "deepseek-r1": (
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}."
        "<｜User｜>{input}<｜Assistant｜><think>\n"
    ),
    "deepseek-r1-choice": (   # For multiple choice question
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{input}<｜Assistant｜><think>\n"
    ),
    "qwen3-think": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ),
    "qwen3-think-choice": (
        "<|im_start|>user\nAnswer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        "{input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ),
    "qwen3-no-think": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    ),
    "qwen3-no-think-choice": (
        "<|im_start|>user\nAnswer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        "{input} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    ),
    "qwen2-5": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "qwen2-5-choice": (
        "<|im_start|>system\nAnswer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama3-1": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "llama3-1-choice": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "openreasoning": (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\nSolve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ),
    "openreasoning-choice": (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\nAnswer the following multiple choice question. The last line of your response should be in the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. (e.g. 'ANSWER: A').\n\n{input}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ),
    "gptoss": (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-09-22\n"
        "\nReasoning: high\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
        "Calls to these tools must go to the commentary channel: 'functions'.<|end|>"
        "<|start|>developer<|message|># Instructions\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|end|>"
        "<|start|>user<|message|>\n{input}\n<|end|><|start|>assistantanalysis"
    ),
    "gptoss_choice": (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-09-22\n"
        "\nReasoning: high\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
        "Calls to these tools must go to the commentary channel: 'functions'.<|end|>"
        "<|start|>developer<|message|># Instructions\n\nAnswer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.<|end|>"
        "<|start|>user<|message|>\n{input}\n<|end|><|start|>assistantanalysis"
    )
}


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def load_data(data_name, data_dir="./datas"):
    data_file = f"{data_dir}/{data_name}/test.jsonl"
    assert os.path.exists(data_file)
    examples = list(load_jsonl(data_file))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def construct_prompt(example, args):
    input_template = PROMPT_TEMPLATES[args.prompt_type]
    full_prompt = input_template.format(input=example["question"])
    return full_prompt