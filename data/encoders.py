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
with open(raw_dir / "shorted_data.jsonl", "r", encoding="utf-8") as f:
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
        print(f"🔄 Đang xử lý dòng {idx}/{total_lines}...")

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
        except json.JSONDecodeError as e:
            continue

        if isinstance(obj, dict) and "input" and "response" in obj:
            input = obj["input"].strip()
            response = obj["response"].strip()
            if input and response:
                finetune_dataset.append({"input": input, "response": response})
            
input, response, input_lengths, response_lengths = [], [], [], []
total_lines = len(finetune_dataset)
for idx, line in enumerate(finetune_dataset):
    if idx % 10000 == 0:
        print(f"🔄 Đang xử lý dòng {idx}/{total_lines}...")

    input_tokens = tokenizer.encode(line["input"]).ids
    response_tokens = tokenizer.encode(line["response"]).ids

    inp = [vocab["[BOS]"]] + input_tokens
    res = input_tokens + [vocab["[SEP]"]] + response_tokens + [vocab["[EOS]"]]

    input.append(inp)
    response.append(res)
    input_lengths.append(len(inp))
    response_lengths.append(len(res))

np.savez_compressed(
    processed_dir / "finetune_data_ids.npz",
    input = np.array(input, dtype=object),
    response = np.array(response, dtype=object),
    input_lengths = np.array(input_lengths, dtype=np.int32),
    response_lengths = np.array(response_lengths, dtype=np.int32)
)
print(f"✅ Đã lưu dữ liệu vào: {processed_dir}/finetune_data_ids.npz")

X, Y, lengths = [], [], []
total_lines = len(pretrain_dataset)
for idx, line in enumerate(pretrain_dataset):
    if idx % 10000 == 0:
        print(f"🔄 Đang xử lý dòng {idx}/{total_lines}...")

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
    processed_dir / "pretrain_data_shorted_ids.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)
print(f"✅ Đã lưu dữ liệu vào: {processed_dir}/pretrain_data_shorted_ids.npz")
