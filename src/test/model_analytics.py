import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import json
import sys
from pathlib import Path
from tokenizers import Tokenizer

from model import TransformerModel

# =====================
# PATH & CONFIG
# =====================
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent
sys.path.append(str(project_root))

config_file = project_root / "config" / "config.json"
model_file = project_root / "model" / "pretrained.pt"
tokenizer_file = project_root / "data" / "processed" / "tokenizer.json"

with open(config_file, "r") as f:
    config = json.load(f)

vocab_size  = config["vocab_size"]
max_seq_len = config["max_seq_len"]
d_model     = config["d_model"]
num_heads   = config["num_heads"]
num_layers  = config["num_layers"]
ff_dim      = config["ff_dim"]
dropout     = config["dropout"]

device = "mps:0"

tokenizer = Tokenizer.from_file(str(tokenizer_file))

def encode(text):
    enc = tokenizer.encode(text)
    return torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0)

model = TransformerModel(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_dim=ff_dim,
    dropout=dropout
).to(device)

state = torch.load(model_file, map_location=device)
model.load_state_dict(state)
model.eval()

# =====================
# 1. BASIC MODEL INFO
# =====================
def log_model_basic(model):
    print("\n========== MODEL BASIC INFO ==========")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Num layers       : {num_layers}")
    print(f"Num heads        : {num_heads}")
    print(f"d_model          : {d_model}")
    print(f"ff_dim           : {ff_dim}")
    print("=====================================")

log_model_basic(model)

# =====================
# 2. ATTENTION HOOKS
# =====================
attn_logs = []

def make_attn_hook(layer_idx):
    def hook(module, input, output):
        # Lấy x từ input
        x = input[0]
        batch_size, seq_len, _ = x.shape
        
        # Tính lại q, k để lấy attention weights
        with torch.no_grad():
            q = module.wq(x)
            k = module.wk(x)
            
            q = q.view(batch_size, seq_len, module.num_heads, module.d_k)
            k = k.view(batch_size, seq_len, module.num_heads, module.d_k)
            
            # Apply RoPE
            cos_freqs, sin_freqs = module.rope(seq_len)
            cos_freqs = cos_freqs.view(1, seq_len, 1, module.d_k)
            sin_freqs = sin_freqs.view(1, seq_len, 1, module.d_k)
            
            q = module.rope.apply_rope(q, cos_freqs, sin_freqs)
            k = module.rope.apply_rope(k, cos_freqs, sin_freqs)
            
            q = q.transpose(1, 2)  # [batch, num_heads, seq_len, d_k]
            k = k.transpose(1, 2)
            
            # Tính attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(module.d_k, dtype=torch.float32))
            
            # Apply causal mask
            causal_mask = module.causal_mask[:seq_len, :seq_len]
            scores = scores + causal_mask
            
            # Softmax to get attention weights
            attn_weights = torch.softmax(scores, dim=-1)
            
            attn_logs.append({
                "layer": f"layer_{layer_idx}",
                "weights": attn_weights.detach().cpu()
            })
    
    return hook

# Register hooks cho tất cả các MultiHeadAttention layers
from model import MultiHeadAttention
for idx, block in enumerate(model.decoder_blocks):
    block.mha.register_forward_hook(make_attn_hook(idx))

# =====================
# 3. RUN SAMPLE INPUT
# =====================
text = "trần cảnh là hoàng đế thái tông"
input_ids = encode(text).to(device)

tokens = tokenizer.encode(text).tokens

print(f"\n========== INPUT TEXT ==========")
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Num tokens: {len(tokens)}")
print("================================")

with torch.no_grad():
    _ = model(input_ids)

# =====================
# 4. LOG HEAD STATISTICS
# =====================
def log_head_stats(attn_logs):
    print("\n========== HEAD STATISTICS ==========")
    for item in attn_logs:
        layer = item["layer"]
        w = item["weights"][0]  # (heads, seq, seq)

        print(f"\n--- {layer} ---")
        for h in range(w.shape[0]):
            entropy = -(w[h] * (w[h] + 1e-9).log()).sum(dim=-1).mean()
            max_attn = w[h].max().item()
            avg_attn = w[h].mean().item()
            print(
                f"Head {h:02d} | "
                f"entropy={entropy:.3f} | "
                f"max_attn={max_attn:.3f} | "
                f"avg_attn={avg_attn:.3f}"
            )

log_head_stats(attn_logs)

# =====================
# 5. TOKEN → TOKEN FOCUS
# =====================
def log_token_focus(attn_logs, tokens, topk=3):
    print("\n========== TOKEN FOCUS (Top 3) ==========")
    for item in attn_logs:
        layer = item["layer"]
        w = item["weights"][0]  # (heads, seq, seq)

        print(f"\n{'='*60}")
        print(f"{layer}")
        print(f"{'='*60}")
        
        for h in range(w.shape[0]):
            print(f"\n{'─'*60}")
            print(f"Head {h}")
            print(f"{'─'*60}")
            for i, tok in enumerate(tokens):
                values = w[h, i]
                top = values.topk(topk)
                targets = [
                    f"{tokens[j]}({values[j]:.2f})"
                    for j in top.indices.tolist()
                ]
                print(f"{tok:>15} -> {', '.join(targets)}")

log_token_focus(attn_logs, tokens)

# =====================
# 6. AUTO SELECT + VISUALIZE ATTENTION (SEMANTIC CRITERIA)
# =====================
def visualize_attention_pattern(attn_logs, tokens, layer_idx, head_idx):
    print(f"\n========== ATTENTION PATTERN ==========")
    print(f"Layer {layer_idx}, Head {head_idx}")
    print("=" * 60)

    w = attn_logs[layer_idx]["weights"][0][head_idx]  # (T, T)

    # Header
    header = "From \\ To"
    print(f"{header:>15}", end="")

    for tok in tokens:
        print(f"{tok:>8}", end="")
    print()
    print("─" * (15 + 8 * len(tokens)))

    # Rows
    for i, tok_from in enumerate(tokens):
        print(f"{tok_from:>15}", end="")
        for j in range(len(tokens)):
            val = w[i, j].item()
            if abs(val) < 1e-6:
                print(f"\033[90m{val:>8.2f}\033[0m", end="")
            elif val > 0.3:
                print(f"\033[91m{val:>8.2f}\033[0m", end="")
            elif val > 0.1:
                print(f"\033[93m{val:>8.2f}\033[0m", end="")
            else:
                print(f"{val:>8.2f}", end="")
        print()


def auto_select_semantic_heads(
    attn_logs,
    tokens,
    top_k=3,
    min_score=0.05,
):
    """
    Chọn head sao cho:
    - 'đế' chú ý mạnh vào ['trần', 'cảnh', 'hoàng']
    - 'tông' chú ý mạnh vào ['thái', 'cảnh', 'trần']
    Score = mean(đế_targets) * mean(tông_targets)
    """

    token_to_idx = {tok: i for i, tok in enumerate(tokens)}

    required_tokens = ["đế", "tông"]
    for tok in required_tokens:
        if tok not in token_to_idx:
            raise ValueError(f"Token '{tok}' không tồn tại trong tokens")

    idx_đế = token_to_idx["đế"]
    idx_tông = token_to_idx["tông"]

    targets_đế = ["trần", "cảnh", "hoàng"]
    targets_tông = ["thái", "cảnh", "trần"]

    idx_targets_đế = [token_to_idx[t] for t in targets_đế if t in token_to_idx]
    idx_targets_tông = [token_to_idx[t] for t in targets_tông if t in token_to_idx]

    results = []

    for layer_idx, item in enumerate(attn_logs):
        # w shape: [B, H, T, T]
        w_all = item["weights"][0]

        num_heads = w_all.shape[0]

        for h in range(num_heads):
            w = w_all[h]

            # --- attention từ token nguồn ---
            attn_đế = w[idx_đế]
            attn_tông = w[idx_tông]

            # --- loại self-attention ---
            attn_đế = attn_đế.clone()
            attn_tông = attn_tông.clone()
            attn_đế[idx_đế] = 0.0
            attn_tông[idx_tông] = 0.0

            # --- tính mean attention đúng target ---
            score_đế = attn_đế[idx_targets_đế].mean().item()
            score_tông = attn_tông[idx_targets_tông].mean().item()

            # --- score semantic (bắt buộc cả hai) ---
            score = score_đế * score_tông

            if score >= min_score:
                results.append({
                    "layer": layer_idx,
                    "head": h,
                    "score": score,
                    "đế": score_đế,
                    "tông": score_tông,
                })

    # sort theo score giảm dần
    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n========== AUTO SELECTED HEADS (SEMANTIC) ==========")
    for i, r in enumerate(results[:top_k]):
        print(
            f"[{i}] Layer {r['layer']} | Head {r['head']} | "
            f"score={r['score']:.3f} | "
            f"đế={r['đế']:.3f} | "
            f"tông={r['tông']:.3f}"
        )

    return results[:top_k]

selected_heads = auto_select_semantic_heads(
    attn_logs,
    tokens,
    top_k=2,
    min_score=0.02,
)

for item in selected_heads:
    visualize_attention_pattern(
        attn_logs,
        tokens,
        layer_idx=item["layer"],
        head_idx=item["head"],
    )
