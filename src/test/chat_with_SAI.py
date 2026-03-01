import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import json
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
from tokenizers import Tokenizer
from src.model import TransformerModel
from src.utils.utils import render_chat_box

config_file = project_root / "config" / "config.json"
sft1_file = project_root / "model" / "sft1.pt"
sft2_file = project_root / "model" / "sft2.pt"
processed_dir = project_root / "data" / "processed"

with open(config_file, 'r') as f:
    config = json.load(f)
vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️  Sử dụng device: {device}")

tokenizer_file = processed_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()

USER = vocab["<|user|>"]
SAI = vocab["<|s.a.i|>"]
BOS = vocab["[BOS]"]
EOS = vocab["[EOS]"]
PAD = vocab["[PAD]"]

def load_model(model_file):
    """Load một model từ file checkpoint"""
    model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    return model

print("\n📦 Đang load các models...")
models = {
    # "SFT1": {"model": load_model(sft1_file)},
    "SAI": {"model": load_model(sft2_file)}
}
print(f"✅ Đã load {len(models)} models: {', '.join(models.keys())}")

def pad_sequence(sequence, max_len, padding_value=0):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [padding_value] * (max_len - len(sequence))

# def generate_response(model, user_input, max_new_tokens=50, beam_size=10):
#     model.eval()
#     prompt = " Input: " + user_input
#     prompt_ids = tokenizer.encode(prompt).ids
#     input = [BOS] + [USER] + prompt_ids + [SAI]
#     output_start_idx = len(input)

#     beams = [{"seq": list(input), "log_prob": 0.0, "done": False}]
#     completed_beams = []

#     def normalized_score(beam):
#         out_len = max(len(beam["seq"]) - output_start_idx, 1)
#         return beam["log_prob"] / (out_len ** 1.0)

#     for step in range(max_new_tokens):
#         active_beams = [b for b in beams if not b["done"]]
#         if not active_beams:
#             break

#         batch_inputs = [pad_sequence(b["seq"], max_seq_len, padding_value=PAD) for b in active_beams]
#         batch_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)

#         with torch.no_grad():
#             logits_batch = model(batch_tensor)

#         all_candidates = []
#         for b_idx, beam in enumerate(active_beams):
#             cur_pos = len(beam["seq"]) - 1
#             logits = logits_batch[b_idx, cur_pos, :]
#             log_probs = torch.log_softmax(logits, dim=-1)
#             log_probs = torch.clamp(log_probs, -1e9, 0.0)

#             topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)
#             for log_p, token in zip(topk_log_probs.cpu().numpy(), topk_tokens.cpu().numpy()):
#                 token = int(token)
#                 new_seq = beam["seq"] + [token]
#                 new_log_prob = beam["log_prob"] + float(log_p)
#                 done = token in [EOS, PAD] or len(new_seq) >= max_seq_len
#                 all_candidates.append({"seq": new_seq, "log_prob": new_log_prob, "done": done})

#         all_candidates.sort(key=normalized_score, reverse=True)
#         kept  = all_candidates[:beam_size]
#         beams = []
#         for cand in kept:
#             if cand["done"]:
#                 completed_beams.append(cand)
#             else:
#                 beams.append(cand)

#         if len(completed_beams) >= beam_size:
#             break

#     final_pool = completed_beams if completed_beams else beams
#     best_beam  = max(final_pool, key=normalized_score)
#     output_tokens = best_beam["seq"][output_start_idx:]
#     while output_tokens and output_tokens[-1] in [EOS, PAD]:
#         output_tokens.pop()

#     return tokenizer.decode(output_tokens)


def generate_response(model, user_input, max_new_tokens=100, beam_size=10, no_repeat_ngram_size=3, repetition_penalty=1.2):
    model.eval()
    prompt = " Input: " + user_input
    prompt_ids = tokenizer.encode(prompt).ids
    input = [BOS] + [USER] + prompt_ids + [SAI]
    output_start_idx = len(input)

    beams = [{"seq": list(input), "log_prob": 0.0, "done": False}]
    completed_beams = []

    def get_banned_tokens(seq):
        """Lấy danh sách token bị cấm do tạo ra n-gram lặp lại."""
        banned = set()
        if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size:
            # Lấy (n-1) token cuối làm prefix để kiểm tra
            ngram_prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
            for i in range(len(seq) - no_repeat_ngram_size + 1):
                if tuple(seq[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned.add(seq[i + no_repeat_ngram_size - 1])
        return banned

    def apply_repetition_penalty(log_probs, seq):
        """Phạt các token đã xuất hiện trong chuỗi hiện tại."""
        if repetition_penalty == 1.0:
            return log_probs
        log_probs = log_probs.clone()
        unique_tokens = set(seq)
        for token_id in unique_tokens:
            if log_probs[token_id] < 0:
                # log_prob âm → nhân penalty làm âm hơn (phạt nặng hơn)
                log_probs[token_id] *= repetition_penalty
            else:
                log_probs[token_id] /= repetition_penalty
        return log_probs

    def normalized_score(beam):
        out_len = max(len(beam["seq"]) - output_start_idx, 1)
        return beam["log_prob"] / (out_len ** 1.0)

    for step in range(max_new_tokens):
        active_beams = [b for b in beams if not b["done"]]
        if not active_beams:
            break

        batch_inputs = [pad_sequence(b["seq"], max_seq_len, padding_value=PAD) for b in active_beams]
        batch_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)

        with torch.no_grad():
            logits_batch = model(batch_tensor)

        all_candidates = []
        for b_idx, beam in enumerate(active_beams):
            cur_pos = len(beam["seq"]) - 1
            logits = logits_batch[b_idx, cur_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.clamp(log_probs, -1e9, 0.0)

            # Áp dụng repetition penalty
            log_probs = apply_repetition_penalty(log_probs, beam["seq"])

            # Lấy danh sách token bị cấm (no_repeat_ngram)
            banned_tokens = get_banned_tokens(beam["seq"])

            topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size + len(banned_tokens))
            count = 0
            for log_p, token in zip(topk_log_probs.cpu().numpy(), topk_tokens.cpu().numpy()):
                if count >= beam_size:
                    break
                token = int(token)

                # Bỏ qua token bị cấm
                if token in banned_tokens:
                    continue

                new_seq = beam["seq"] + [token]
                new_log_prob = beam["log_prob"] + float(log_p)
                done = token in [EOS, PAD] or len(new_seq) >= max_seq_len
                all_candidates.append({"seq": new_seq, "log_prob": new_log_prob, "done": done})
                count += 1

        all_candidates.sort(key=normalized_score, reverse=True)
        kept  = all_candidates[:beam_size]
        beams = []
        for cand in kept:
            if cand["done"]:
                completed_beams.append(cand)
            else:
                beams.append(cand)

        if len(completed_beams) >= beam_size:
            break

    final_pool = completed_beams if completed_beams else beams
    best_beam  = max(final_pool, key=normalized_score)
    output_tokens = best_beam["seq"][output_start_idx:]
    while output_tokens and output_tokens[-1] in [EOS, PAD]:
        output_tokens.pop()

    return tokenizer.decode(output_tokens)

if __name__ == "__main__":
    test_cases = [
        # "chào",
        # "3+5 bằng mấy",
        # "9+8",
        # "có đó không",
        "max verstappen là ai",
        # "ê sai",
        "formula one là gì",
        "đinh tiên hoàng đế có huý danh là gì",
        "giỏi thế",
        "tên thật của trần thái tổ là gì",
        "trần nhân tông có tên thật là gì",
        # "ê dậy đi",
        # "miếu hiệu của lê lợi",
        # "huý danh của lê thái tông là gì",
        # "lê thánh tông có huý là gì",
        # "miếu hiệu của hoàng đế trần hoảng là gì",
        # "hello sai",
        "Việt Nam có món ăn nào ngon",
        "cách nấu phở",
        "oke cảm ơn"
    ]

    render_chat_box(test_cases, models, generate_response)
