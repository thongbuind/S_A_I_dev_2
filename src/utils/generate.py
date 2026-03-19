import torch
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

def build_input(user_input, tokenizer):
    vocab = tokenizer.get_vocab()
    BOS = vocab["[BOS]"]
    EOS = vocab["[EOS]"]
    PAD = vocab["[PAD]"]
    USER = vocab["<|user|>"]
    SAI = vocab["<|s.a.i|>"]
    prompt_ids = tokenizer.encode(" Input: " + user_input).ids
    input_ids  = [BOS, USER] + prompt_ids + [SAI]
    return input_ids, len(input_ids), EOS, PAD


def decode_output(best, start, EOS, PAD, tokenizer):
    output_tokens = best["seq"][start:]
    while output_tokens and output_tokens[-1] in [EOS, PAD]:
        output_tokens.pop()
    return tokenizer.decode(output_tokens)


# ================= SCORE / PENALTY =================
def score(seq, log_prob, start):
    out_len = max(len(seq) - start, 1)
    return log_prob / (out_len ** 1.0)

def apply_penalty(logits, seq, penalty):
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for tid in set(seq):
        if logits[tid] < 0:
            logits[tid] *= penalty
        else:
            logits[tid] /= penalty
    return logits

# ================= BANNED TOKENS =================
def get_banned_tokens(seq, n):
    banned = set()
    if n > 0 and len(seq) >= n:
        prefix = tuple(seq[-(n - 1):])
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
    return banned

# ================= FORWARD =================
def forward_init(model, input_ids, max_beam_size, max_new_tokens, device):
    prompt_len = len(input_ids)
    max_total  = prompt_len + max_new_tokens

    kv_buffers = [
        (torch.zeros(max_beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device),
         torch.zeros(max_beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device))
        for block in model.decoder_blocks
    ]

    prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        x = model.token_embedding(prompt_tensor)
        x = model.dropout_layer(x)

        for i, block in enumerate(model.decoder_blocks):
            x_out, present_kv = block.mha.prefill(x)
            kv_buffers[i][0][:, :, :prompt_len, :].copy_(present_kv[0].expand(max_beam_size, -1, -1, -1))
            kv_buffers[i][1][:, :, :prompt_len, :].copy_(present_kv[1].expand(max_beam_size, -1, -1, -1))

            attn_out = block.dropout1(x_out)
            out1     = block.layernorm1(x + attn_out)
            ffn_out  = block.ffn(out1)
            ffn_out  = block.dropout2(ffn_out)
            x        = block.layernorm2(out1 + ffn_out)

        logits = model.final_layer(x)[0, -1, :]

    return logits, kv_buffers, prompt_len

def forward_step(model, last_tokens, kv_buffers, cache_len):
    with torch.inference_mode():
        x = model.token_embedding(last_tokens)
        x = model.dropout_layer(x)
        for i, block in enumerate(model.decoder_blocks):
            x = block.forward_with_cache(x, kv_buffers[i], cache_len)
        logits = model.final_layer(x)[:, 0, :]
    return logits

# ================= BEAM CORE =================
def beam_core(model, input_ids, start,
                      score_fn, penalty_fn,
                      max_new_tokens, beam_size,
                      no_repeat_ngram, penalty,
                      device, EOS, PAD, max_seq_len,
                      early_stop=True, patience=10):

    first_logits, kv_buffers, cache_len = forward_init(
        model, input_ids, beam_size, max_new_tokens, device
    )

    first_logits = penalty_fn(first_logits, input_ids, penalty)
    first_lp = torch.clamp(torch.log_softmax(first_logits, -1), -1e9, 0.0)
    topk_lp, topk_tok = torch.topk(first_lp, beam_size)

    seqs = [input_ids + [int(t)] for t in topk_tok.tolist()]
    log_probs = topk_lp.tolist()
    dones = [int(t) in [EOS, PAD] for t in topk_tok.tolist()]
    unique_sets = [set(input_ids) | {int(t)} for t in topk_tok.tolist()]
    completed = []
    K = beam_size * 3
    patience_counter = 0

    for _ in range(max_new_tokens - 1):
        if all(dones):
            break

        last = torch.tensor([[seqs[i][-1]] for i in range(len(seqs))], dtype=torch.long, device=device)
        logits_batch = forward_step(model, last, kv_buffers, cache_len)
        cache_len += 1

        n_beams = len(seqs)
        if penalty != 1.0:
            pen_mask = torch.ones(n_beams, logits_batch.shape[-1], device=device)
            for i, uid in enumerate(unique_sets):
                if dones[i]:
                    continue
                idx = torch.tensor(list(uid), dtype=torch.long, device=device)
                pen_mask[i, idx] = penalty
            neg = logits_batch < 0
            logits_batch = torch.where(neg, logits_batch * pen_mask, logits_batch / pen_mask)

        lp_batch = torch.clamp(torch.log_softmax(logits_batch, -1), -1e9, 0.0)
        topk_lp_b, topk_tok_b = torch.topk(lp_batch, K, dim=-1)
        topk_lp_b = topk_lp_b.tolist()
        topk_tok_b = topk_tok_b.tolist()

        candidates = []
        for beam_i in range(n_beams):
            if dones[beam_i]:
                continue
            banned = get_banned_tokens(seqs[beam_i], no_repeat_ngram)
            count = 0
            for l, t in zip(topk_lp_b[beam_i], topk_tok_b[beam_i]):
                if count >= beam_size:
                    break
                if t in banned:
                    continue
                new_seq = seqs[beam_i] + [t]
                done = t in [EOS, PAD] or len(new_seq) >= max_seq_len
                candidates.append({
                    "beam_src": beam_i,
                    "seq": new_seq,
                    "log_prob": log_probs[beam_i] + l,
                    "done": done
                })
                count += 1

        if not candidates:
            break

        candidates.sort(key=lambda x: score_fn(x["seq"], x["log_prob"], start), reverse=True)

        kept, src = [], []
        for c in candidates:
            if c["done"]:
                completed.append(c)
            elif len(kept) < beam_size:
                kept.append(c)
                src.append(c["beam_src"])

        if not kept:
            break

        if len(completed) >= beam_size:
            best_done = max(completed, key=lambda x: score_fn(x["seq"], x["log_prob"], start))
            best_alive = max(kept, key=lambda x: score_fn(x["seq"], x["log_prob"], start))
            if score_fn(best_done["seq"], best_done["log_prob"], start) >= \
               score_fn(best_alive["seq"], best_alive["log_prob"], start):
                if early_stop:
                    break
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            else:
                patience_counter = 0

        if src != list(range(len(src))):
            src_t = torch.tensor(src, dtype=torch.long, device=device)
            for k_buf, v_buf in kv_buffers:
                k_tmp = k_buf[src_t].clone()
                v_tmp = v_buf[src_t].clone()
                k_buf[:len(src)].copy_(k_tmp)
                v_buf[:len(src)].copy_(v_tmp)

        unique_sets = [unique_sets[src[i]] | {c["seq"][-1]} for i, c in enumerate(kept)]
        seqs = [c["seq"] for c in kept]
        log_probs = [c["log_prob"] for c in kept]
        dones = [False] * len(kept)

    pool = completed if completed else [{"seq": seqs[i], "log_prob": log_probs[i]} for i in range(len(seqs))]
    return max(pool, key=lambda x: score_fn(x["seq"], x["log_prob"], start))


# ================= PUBLIC API =================

def generate(model, user_input, tokenizer,
                   max_new_tokens=200, beam_size=5,
                   no_repeat_ngram=3, penalty=1.2,
                   early_stop=False, patience=15):

    device = model.final_layer.weight.device
    ids, start, EOS, PAD = build_input(user_input, tokenizer)

    best = beam_core(
        model, ids, start,
        score, apply_penalty,
        max_new_tokens, beam_size, no_repeat_ngram, penalty,
        device, EOS, PAD, model.max_seq_len,
        early_stop=early_stop, patience=patience
    )

    return decode_output(best, start, EOS, PAD, tokenizer)
