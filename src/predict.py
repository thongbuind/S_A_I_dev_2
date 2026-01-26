import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import json
import sys
from pathlib import Path
from tokenizers import Tokenizer
from model import TransformerModel

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"
vocab_file = project_root / "data" / "vocab.txt"
model_file = project_root / "model" / "s_a_i.pt"
processed_dir = project_root / "data" / "processed"

# Load config
with open(config_file, 'r') as f:
    config = json.load(f)

vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']

# Load model
device = torch.device('mps:0')
model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer_file = processed_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()
idx2word = {i: w for w, i in vocab.items()}

special_tokens = {
    "PAD": vocab.get("[PAD]", 0),
    "UNK": vocab.get("[UNK]", 1),
    "BOS": vocab.get("[BOS]", 2),
    "EOS": vocab.get("[EOS]", 3),
    "SEP": vocab.get("[SEP]", 4),
}

def tokenize(sentence):
    return tokenizer.encode(sentence.lower()).ids

def detokenize(tokens):
    return tokenizer.decode(tokens)

def pad_sequence(sequence, max_len, padding_value=0):
    """Tương đương tf.keras.preprocessing.sequence.pad_sequences"""
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [padding_value] * (max_len - len(sequence))

def generate_response_0(sentence, max_new_tokens=100, loss_margin=0.2):
    """
    Tạo phản hồi từ câu đầu vào:
        current_sequence = [BOS] + req
        sequence = loop(predict(current_sequence))

    Sampling strategy:
        - Chọn token có loss thấp nhất
        - Tìm các token có loss không vượt quá best_loss + loss_margin
        - Random trong nhóm đó
    """
    req_tokens = tokenize(sentence)
    current_sequence = [vocab["[BOS]"]] + req_tokens

    # Tương đương tf.keras.preprocessing.sequence.pad_sequences
    padded_input = pad_sequence(current_sequence, max_seq_len, padding_value=0)
    padded_input = torch.tensor([padded_input], dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        with torch.no_grad():
            # model(padded_input) tương đương TF
            preds = model(padded_input)[0, len(current_sequence) - 1, :]
            
            # Tương đương tf.nn.softmax
            preds = torch.softmax(preds, dim=-1)
            
            # Tương đương np.clip(preds, 1e-9, 1.0)
            preds = torch.clamp(preds, 1e-9, 1.0)

            # Tương đương token_losses = -np.log(preds)
            token_losses = -torch.log(preds)
            best_loss = torch.min(token_losses)

            # Tìm các token có loss <= best_loss + margin
            candidate_mask = token_losses <= best_loss + loss_margin
            candidate_indices = torch.where(candidate_mask)[0].cpu().numpy()
            
            next_token = np.random.choice(candidate_indices)

        if next_token in [vocab["[EOS]"], vocab["[PAD]"]]:
            break

        current_sequence.append(int(next_token))
        padded_input[0, len(current_sequence) - 1] = next_token

        if len(current_sequence) >= max_seq_len:
            break

    return detokenize(current_sequence[1:])

def generate_response(sentence, max_new_tokens=100, loss_margin=0.2, lookahead=5):
    req_tokens = tokenize(sentence)
    current_sequence = [vocab["[BOS]"]] + req_tokens

    for step in range(max_new_tokens):
        # Pad sequence
        padded_input = pad_sequence(current_sequence, max_seq_len, padding_value=0)
        padded_input = torch.tensor([padded_input], dtype=torch.long, device=device)
        
        with torch.no_grad():
            preds = model(padded_input)[0, len(current_sequence) - 1, :]
            preds = torch.softmax(preds, dim=-1)
            preds = torch.clamp(preds, 1e-9, 1.0)

            token_losses = -torch.log(preds)
            best_loss = torch.min(token_losses)
            margin_mask = token_losses <= best_loss + loss_margin
            candidate_indices = torch.where(margin_mask)[0].cpu().numpy()

        if len(candidate_indices) == 0:
            break

        # Tạo batch inputs cho lookahead
        batch_inputs = []
        for idx in candidate_indices:
            seq = current_sequence + [int(idx)]
            padded_seq = pad_sequence(seq, max_seq_len, padding_value=0)
            batch_inputs.append(padded_seq)

        batch_inputs = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        total_losses = np.zeros(len(candidate_indices))

        # Lookahead loop
        for i in range(lookahead):
            with torch.no_grad():
                preds_batch = model(batch_inputs)
                
                # Lấy predictions tại vị trí hiện tại
                step_index = len(current_sequence) + i - 1
                step_preds = preds_batch[:, step_index, :]
                
                step_preds = torch.softmax(step_preds, dim=-1)
                step_preds = torch.clamp(step_preds, 1e-9, 1.0)
                step_losses = -torch.log(step_preds)

                min_losses = torch.min(step_losses, dim=-1)[0].cpu().numpy()
                total_losses += min_losses

                next_tokens = torch.argmin(step_losses, dim=-1).cpu().numpy()
                
                # Update batch_inputs với next tokens
                for j, token in enumerate(next_tokens):
                    if len(current_sequence) + i < max_seq_len:
                        batch_inputs[j, len(current_sequence) + i] = token

        # Chọn token tốt nhất
        best_total = np.min(total_losses)
        valid_mask = total_losses <= best_total + loss_margin
        valid_indices = np.where(valid_mask)[0]
        chosen_idx = np.random.choice(valid_indices)

        next_token = candidate_indices[chosen_idx]
        if next_token in [vocab["[EOS]"], vocab["[PAD]"]]:
            break

        current_sequence.append(int(next_token))
        if len(current_sequence) >= max_seq_len:
            break

    return detokenize(current_sequence[1:])

# ================
# Kiểm Tra Mô Hình
# ================

inputs = [
    # "xin chào",
    # "chào buổi sáng",
    "hoàng",
    "Lê Lợi",
    "thái tổ cao hoàng đế",
    "Đinh Tiên Hoàng tên thật là",
    "sau khi lên ngôi",
    "nhà trần",
    "nhà lý",
    "triều đại hậu lê",
    "việt nam",
    "việt nam sở hữu",
    "phở",
    "bánh mì",
    "bát bún",
    "giải đua xe công thức 1",
    "max verstappen",
    "leclerc",
    "manchester",
    "vào buổi tối",
    "sau khi ăn xong, chúng tôi",
]

print("\n=== Test pre-train ===")
i = 1
for req in inputs:
    print(f"Req {i}: {req}")
    print(f"Res {i}.prev: {generate_response_0(req)}")
    print(f"Res {i}.current: {generate_response(req)}\n")
    i += 1
