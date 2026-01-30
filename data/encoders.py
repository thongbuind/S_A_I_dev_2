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
idx2word = {i: w for w, i in vocab.items()}

pretrain_dataset = []
with open(raw_dir / "pretrain_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            continue

        if isinstance(obj, dict) and "text" in obj:
            text = obj["text"].strip()
            if text:
                pretrain_dataset.append(text)

continued_pretrain_dataset = []
with open(raw_dir / "continued_pretrain_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            continue

        if isinstance(obj, dict) and "text" in obj:
            text = obj["text"].strip()
            if text:
                continued_pretrain_dataset.append(text)

X, Y, lengths = [], [], []
total_lines = len(continued_pretrain_dataset)
for idx, line in enumerate(continued_pretrain_dataset):
    if idx % 10000 == 0:
        print(f"📄 Đang xử lý dòng {idx}/{total_lines}...")

    encoded = tokenizer.encode(line)
    tokens = encoded.ids

    if len(tokens) < 2 or len(tokens) + 2 > max_seq_len:
        continue

    inp = [vocab["[BOS]"]] + tokens
    tgt = tokens + [vocab["[EOS]"]]

    X.append(inp)
    Y.append(tgt)
    lengths.append(len(inp))

np.savez_compressed(
    processed_dir / "continued_pretrain_data_ids.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)
print(f"✅ Đã lưu dữ liệu vào: {processed_dir}/continued_pretrain_data_ids.npz")

finetune_dataset = []
with open(raw_dir / "finetune_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        instruction = obj.get("instruction", "").strip()
        input_text = obj.get("input", "").strip()
        output_text = obj.get("output", "").strip()

        if not instruction or not output_text:
            continue

        finetune_dataset.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

X, Y, loss_mask, lengths = [], [], [], []
total_lines = len(finetune_dataset)

USER = vocab["<|user|>"]
SAI = vocab["<|s.a.i|>"]
BOS = vocab["[BOS]"]
EOS = vocab["[EOS]"]

for idx, sample in enumerate(finetune_dataset):
    if idx % 10000 == 0:
        print(f"📄 Đang xử lý dòng {idx}/{total_lines}...")

    if sample["input"]:
        prompt = (
            "Instruction: " + sample["instruction"] +
            " Input: " + sample["input"]
        )
    else:
        prompt = "Instruction: " + sample["instruction"]

    prompt_ids = tokenizer.encode(prompt).ids
    output_ids = tokenizer.encode(sample["output"]).ids

    input_ids = ([BOS] + [USER] + prompt_ids + [SAI] + output_ids + [EOS])

    if len(input_ids) > max_seq_len:
        continue

    target_ids = input_ids[1:]

    mask = (
        [0] * (1 + len(prompt_ids) + 1) +  # USER + prompt + SAI
        [1] * (len(output_ids) + 1)         # output + EOS
    )
    
    assert len(mask) == len(target_ids), \
        f"Mask length mismatch: mask={len(mask)}, target={len(target_ids)}"

    X.append(input_ids)
    Y.append(target_ids)
    loss_mask.append(mask)
    lengths.append(len(input_ids))

np.savez_compressed(
    processed_dir / "finetune_data_ids.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    loss_mask=np.array(loss_mask, dtype=object),
    lengths=np.array(lengths, dtype=np.int32)
)

print(f"✅ Đã lưu dữ liệu finetune đúng format vào: {processed_dir}/finetune_data_ids.npz")

X, Y, lengths = [], [], []
total_lines = len(pretrain_dataset)
for idx, line in enumerate(pretrain_dataset):
    if idx % 10000 == 0:
        print(f"📄 Đang xử lý dòng {idx}/{total_lines}...")

    encoded = tokenizer.encode(line)
    tokens = encoded.ids

    if len(tokens) < 2 or len(tokens) + 2 > max_seq_len:
        continue

    inp = [vocab["[BOS]"]] + tokens
    tgt = tokens + [vocab["[EOS]"]]

    X.append(inp)
    Y.append(tgt)
    lengths.append(len(inp))

print(f"\n📊 THỐNG KÊ DỮ LIỆU:")
print(f"📊 Tổng số mẫu: {len(X)}")
print(f"📈 Độ dài sequence trung bình: {np.mean(lengths):.2f}")
print(f"📉 Độ dài sequence min/max: {min(lengths)}/{max(lengths)}")

np.savez_compressed(
    processed_dir / "pretrain_data_ids.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)
print(f"✅ Đã lưu dữ liệu vào: {processed_dir}/pretrain_data_ids.npz")
