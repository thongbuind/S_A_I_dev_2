import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

current_file = Path(__file__).resolve()
data_dir = current_file.parent
config_file = data_dir.parent / "config" / "config.json"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

with open(config_file, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

tokenizer_file = processed_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()

def load_text_jsonl(path, key="text"):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(obj, dict) and key in obj:
                text = obj[key].strip()
                if text: data.append(text)
    return data

def load_sft_jsonl(path):
    dataset = []
    with open(raw_dir / path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError: continue
            if not isinstance(obj, dict): continue

            instruction = obj.get("instruction", "").strip()
            input_text = obj.get("input", "").strip()
            output_text = obj.get("output", "").strip()

            if not output_text: continue

            dataset.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
    return dataset

def build_dataset(texts, tokenizer, vocab, max_seq_len):
    X, Y, lengths = [], [], []
    for idx, line in enumerate(texts):
        if idx % 10000 == 0:
            print(f"📄 Đang xử lý dòng {idx}/{len(texts)}...")
        tokens = tokenizer.encode(line).ids
        if len(tokens) < 2 or len(tokens) + 2 > max_seq_len:
            continue
        X.append([vocab["[BOS]"]] + tokens)
        Y.append(tokens + [vocab["[EOS]"]])
        lengths.append(len(tokens) + 1)
    return X, Y, lengths

pretrain_texts = load_text_jsonl(raw_dir / "pretrain_data.jsonl")
X, Y, lengths = build_dataset(pretrain_texts, tokenizer, vocab, max_seq_len)
np.savez_compressed(processed_dir / "pretrain_data_ids.npz", X=np.array(X, dtype=object), Y=np.array(Y, dtype=object), lengths=np.array(lengths))

continued_texts = load_text_jsonl(raw_dir / "continued_pretrain_data.jsonl")
X, Y, lengths = build_dataset(continued_texts, tokenizer, vocab, max_seq_len)
np.savez_compressed(processed_dir / "continued_pretrain_data_ids.npz", X=np.array(X, dtype=object), Y=np.array(Y, dtype=object), lengths=np.array(lengths))

USER = vocab["<|user|>"]
SAI = vocab["<|s.a.i|>"]
BOS = vocab["[BOS]"]
EOS = vocab["[EOS]"]

def process_sft_data(dataset, use_fixed_instruction=False):
    X, Y, loss_mask, lengths = [], [], [], []
    
    for idx, sample in enumerate(dataset):
        if idx % 100 == 0:
            print(f"📄 Đang xử lý dòng {idx}/{len(dataset)}...")

        if use_fixed_instruction:
            instruction = "Trả lời input sau bằng tiếng Việt"
        else:
            instruction = sample["instruction"]
        
        if sample["input"]:
            prompt = "Instruction: " + instruction + " Input: " + sample["input"]
        else:
            prompt = "Instruction: " + instruction

        prompt_ids = tokenizer.encode(prompt).ids
        output_ids = tokenizer.encode(sample["output"]).ids

        input_ids = ([BOS] + [USER] + prompt_ids + [SAI] + output_ids + [EOS])

        if len(input_ids) > max_seq_len:
            continue

        target_ids = input_ids[1:]

        mask = (
            [0] * (1 + len(prompt_ids) + 1) +
            [1] * (len(output_ids) + 1)
        )
        
        assert len(mask) == len(target_ids)

        X.append(input_ids)
        Y.append(target_ids)
        loss_mask.append(mask)
        lengths.append(len(input_ids))
    
    return X, Y, loss_mask, lengths

sft1_dataset = load_sft_jsonl("SFT_1.jsonl")
X_sft1, Y_sft1, loss_mask_sft1, lengths_sft1 = process_sft_data(sft1_dataset, use_fixed_instruction=True)

np.savez_compressed(
    processed_dir / "SFT1_data_ids.npz",
    X=np.array(X_sft1, dtype=object),
    Y=np.array(Y_sft1, dtype=object),
    loss_mask=np.array(loss_mask_sft1, dtype=object),
    lengths=np.array(lengths_sft1, dtype=np.int32)
)
print(f"✅ Đã lưu SFT1: {len(X_sft1)} samples")

sft2_dataset = load_sft_jsonl("SFT_2.jsonl")
X_sft2, Y_sft2, loss_mask_sft2, lengths_sft2 = process_sft_data(sft2_dataset, use_fixed_instruction=False)

np.savez_compressed(
    processed_dir / "SFT2_data_ids.npz",
    X=np.array(X_sft2, dtype=object),
    Y=np.array(Y_sft2, dtype=object),
    loss_mask=np.array(loss_mask_sft2, dtype=object),
    lengths=np.array(lengths_sft2, dtype=np.int32)
)
print(f"✅ Đã lưu SFT2: {len(X_sft2)} samples")
