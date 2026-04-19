# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# import torch
# import json
# import time
# import sys
# from pathlib import Path
# current_file = Path(__file__).resolve()
# src_dir = current_file.parent.parent
# project_root = src_dir.parent
# sys.path.append(str(project_root))
# from tokenizers import Tokenizer
# from src.model import TransformerModel

# config_dir = project_root / "config"
# base_config_file = config_dir / "base.json"
# model_config_file = config_dir / f"35M.json"

# sft1_file = project_root / "model" / "sft1_35M.pt"
# data_dir = project_root / "data"

# with open(base_config_file, 'r') as f:
#     config = json.load(f)
# with open(model_config_file, 'r') as f:
#     config.update(json.load(f))
# vocab_size = config['vocab_size']
# max_seq_len = config['max_seq_len']
# d_model = config['d_model']
# num_heads = config['num_heads']
# num_layers = config['num_layers']
# ff_dim = config['ff_dim']
# dropout = config['dropout']

# device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
# print(f"🖥️  Sử dụng device: {device}")

# tokenizer_file = data_dir / "tokenizer.json"
# tokenizer = Tokenizer.from_file(str(tokenizer_file))
# vocab = tokenizer.get_vocab()

# USER = vocab["<|user|>"]
# SAI = vocab["<|s.a.i|>"]
# BOS = vocab["[BOS]"]
# EOS = vocab["[EOS]"]
# PAD = vocab["[PAD]"]

# def load_model(model_file):
#     model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
#     model.load_state_dict(torch.load(model_file, map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# models = {
#     "sft1": {"model": load_model(sft1_file)}
# }
# print(f"✅ Đã load {len(models)} models: {', '.join(models.keys())}")
# total_params = 0
# for param_name, param in models["sft1"]["model"].named_parameters():
#     total_params += param.numel()
        
# print(f"👉 Tổng số tham số của model: {total_params:,}")

# def pad_sequence(sequence, max_len, padding_value=0):
#     if len(sequence) >= max_len:
#         return sequence[:max_len]
#     return sequence + [padding_value] * (max_len - len(sequence))

# def generate_response_prev(model, user_input, max_new_tokens=50, beam_size=5):
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

#         attention_mask = (batch_tensor != PAD).long()

#         with torch.inference_mode():
#             logits_batch = model(batch_tensor, attention_mask=attention_mask)

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

#                 all_candidates.append({
#                     "seq": new_seq,
#                     "log_prob": new_log_prob,
#                     "done": done
#                 })

#         all_candidates.sort(key=normalized_score, reverse=True)
#         kept = all_candidates[:beam_size]

#         beams = []
#         for cand in kept:
#             if cand["done"]:
#                 completed_beams.append(cand)
#             else:
#                 beams.append(cand)

#         if len(completed_beams) >= beam_size:
#             break

#     final_pool = completed_beams if completed_beams else beams
#     best_beam = max(final_pool, key=normalized_score)

#     output_tokens = best_beam["seq"][output_start_idx:]
#     while output_tokens and output_tokens[-1] in [EOS, PAD]:
#         output_tokens.pop()

#     return tokenizer.decode(output_tokens)

# test_cases = [
#     "formula 1 là gì",
#     "tôi muốn tìm hiểu về giải đua công thức 1",
#     "formula one là giải đua gì",
#     "oracle redbull racing",
#     "redbull racing",
#     "mercedes-amg petronat formula one team",
#     "cách làm bánh mì việt nam",
#     "cho tôi thêm thông tin về đội đua f1 ferrari",
#     "max verstappen là ai",
#     "ẩm thực việt nam",
#     "cách nấu phở",
#     "cách làm pizza",
#     "hướng dẫn tôi làm một ly sinh tố dâu tây thật mát lạnh trong ngày hè nóng bức",
#     "tôi muốn làm một lý sinh tố bơ",
#     "hướng dẫn cho tôi cách nấu cháo gà",
#     "làm hamburger tại nhà",
#     "manchester united",
#     "sir alex Ferguson là ai",
#     "sir alex đã vô địch ngoại hạng anh mấy lần",
#     "kim dahyun là ai",
#     "twice gồm mấy thành viên",
#     "huyết áp trung bình (MAP) là gì",
#     "kỹ thuật xoáy xuống trong bóng bàn",
#     "thông tin về bts",
#     "bts có mấy thành viên",
#     "sơn tùng m-tp là ai"
# ]

# # ================= COMMON =================
# def build_input(user_input):
#     prompt = " Input: " + user_input
#     prompt_ids = tokenizer.encode(prompt).ids
#     input_ids = [BOS] + [USER] + prompt_ids + [SAI]
#     return input_ids, len(input_ids)

# def get_banned_tokens(seq, n):
#     banned = set()
#     if n > 0 and len(seq) >= n:
#         prefix = tuple(seq[-(n - 1):])
#         for i in range(len(seq) - n + 1):
#             if tuple(seq[i:i + n - 1]) == prefix:
#                 banned.add(seq[i + n - 1])
#     return banned

# def apply_penalty(logits, seq, penalty):
#     if penalty == 1.0:
#         return logits
#     logits = logits.clone()
#     for tid in set(seq):
#         if logits[tid] < 0:
#             logits[tid] *= penalty
#         else:
#             logits[tid] /= penalty
#     return logits

# def score_huong_1(seq, log_prob, start):
#     out_len = max(len(seq) - start, 1)
#     return log_prob / (out_len ** 1.0)

# def score_huong_2(seq, log_prob, start):
#     out_len = max(len(seq) - start, 1)
#     lp = ((5 + out_len) / 6) ** 0.6
#     return log_prob / lp

# # ================= FORWARD =================
# def forward_huong_1(model, beams):
#     max_len = max(len(b["seq"]) for b in beams)
#     batch_inputs = [pad_sequence(b["seq"], max_len, PAD) for b in beams]
#     batch_tensor = torch.as_tensor(batch_inputs, dtype=torch.long, device=device)
#     attention_mask = (batch_tensor != PAD).long()

#     with torch.inference_mode():
#         logits_batch = model(batch_tensor, attention_mask=attention_mask)

#     out = []
#     for i, b in enumerate(beams):
#         cur_pos = len(b["seq"]) - 1
#         out.append(logits_batch[i, cur_pos, :])
#     return out

# def forward_huong_2_init(model, input_ids, max_beam_size, max_new_tokens):
#     prompt_len = len(input_ids)
#     max_total = prompt_len + max_new_tokens

#     kv_buffers = []
#     for block in model.decoder_blocks:
#         k_buf = torch.zeros(max_beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device)
#         v_buf = torch.zeros(max_beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device)
#         kv_buffers.append((k_buf, v_buf))

#     prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

#     with torch.inference_mode():
#         first_logits, present_cache = model.prefill(prompt_tensor, kv_cache=None)
#         for i, (k, v) in enumerate(present_cache):
#             kv_buffers[i][0][:, :, :prompt_len, :].copy_(k.expand(max_beam_size, -1, -1, -1))
#             kv_buffers[i][1][:, :, :prompt_len, :].copy_(v.expand(max_beam_size, -1, -1, -1))

#     return first_logits[0], kv_buffers, prompt_len

# def forward_huong_2_step(model, last_tokens_full, kv_buffers, cache_len):
#     with torch.inference_mode():
#         x = model.token_embedding(last_tokens_full) 
#         x = model.dropout_layer(x)

#         for l, block in enumerate(model.decoder_blocks):
#             x = block.forward_with_cache(x, kv_buffers[l], cache_len)

#         logits = model.final_layer(model.final_norm(x))[:, 0, :]

#     return logits

# # ================= CORE 1 (no cache) =================
# def beam_core_huong_1(model, input_ids, start,
#                      score_fn, penalty_fn,
#                      max_new_tokens, beam_size,
#                      no_repeat_ngram, penalty):

#     beams = [{"seq": list(input_ids), "log_prob": 0.0, "done": False}]
#     completed = []

#     for _ in range(max_new_tokens):
#         active = [b for b in beams if not b["done"]]
#         if not active:
#             break

#         logits_list = forward_huong_1(model, active)

#         candidates = []
#         for beam, logits in zip(active, logits_list):
#             logits = penalty_fn(logits, beam["seq"], penalty)

#             lp = torch.clamp(torch.log_softmax(logits, -1), -1e9, 0.0)
#             banned = get_banned_tokens(beam["seq"], no_repeat_ngram)

#             topk_lp, topk_tok = torch.topk(lp, beam_size * 3)

#             count = 0
#             for l, t in zip(topk_lp.tolist(), topk_tok.tolist()):
#                 if count >= beam_size:
#                     break
#                 if t in banned:
#                     continue

#                 new_seq = beam["seq"] + [t]
#                 done = t in [EOS, PAD] or len(new_seq) >= max_seq_len

#                 candidates.append({
#                     "seq": new_seq,
#                     "log_prob": beam["log_prob"] + l,
#                     "done": done
#                 })
#                 count += 1

#         candidates.sort(key=lambda x: score_fn(x["seq"], x["log_prob"], start), reverse=True)

#         beams = []
#         for c in candidates[:beam_size]:
#             if c["done"]:
#                 completed.append(c)
#             else:
#                 beams.append(c)

#         if len(completed) >= beam_size:
#             break

#     pool = completed if completed else beams
#     return max(pool, key=lambda x: score_fn(x["seq"], x["log_prob"], start))

# # ================= CORE 2 (kv cache) =================
# def beam_core_huong_2(model, input_ids, start,
#                      score_fn, penalty_fn,
#                      max_new_tokens, beam_size,
#                      no_repeat_ngram, penalty,
#                      early_stop=True, patience=10):

#     first_logits, kv_buffers, cache_len = forward_huong_2_init(
#         model, input_ids, beam_size, max_new_tokens
#     )

#     first_logits = penalty_fn(first_logits, input_ids, penalty)
#     first_lp = torch.clamp(torch.log_softmax(first_logits, -1), -1e9, 0.0)
#     topk_lp, topk_tok = torch.topk(first_lp, beam_size)

#     seqs        = [input_ids + [int(t)] for t in topk_tok.tolist()]
#     log_probs   = topk_lp.tolist()
#     dones       = [int(t) in [EOS, PAD] for t in topk_tok.tolist()]
#     unique_sets = [set(input_ids) | {int(t)} for t in topk_tok.tolist()]
#     completed   = []
#     K           = beam_size * 3
#     patience_counter = 0

#     for _ in range(max_new_tokens - 1):
#         if all(dones):
#             break

#         last = torch.tensor([[seqs[i][-1]] for i in range(len(seqs))], dtype=torch.long, device=device)
#         logits_batch = forward_huong_2_step(model, last, kv_buffers, cache_len)
#         cache_len += 1

#         n_beams = len(seqs)
#         if penalty != 1.0:
#             pen_mask = torch.ones(n_beams, logits_batch.shape[-1], device=device)
#             for i, uid in enumerate(unique_sets):
#                 if dones[i]:
#                     continue
#                 idx = torch.tensor(list(uid), dtype=torch.long, device=device)
#                 pen_mask[i, idx] = penalty
#             neg = logits_batch < 0
#             logits_batch = torch.where(neg, logits_batch * pen_mask, logits_batch / pen_mask)

#         lp_batch = torch.clamp(torch.log_softmax(logits_batch, -1), -1e9, 0.0)
#         topk_lp_b, topk_tok_b = torch.topk(lp_batch, K, dim=-1)
#         topk_lp_b  = topk_lp_b.tolist()
#         topk_tok_b = topk_tok_b.tolist()

#         candidates = []
#         for beam_i in range(n_beams):
#             if dones[beam_i]:
#                 continue
#             banned = get_banned_tokens(seqs[beam_i], no_repeat_ngram)
#             count = 0
#             for l, t in zip(topk_lp_b[beam_i], topk_tok_b[beam_i]):
#                 if count >= beam_size:
#                     break
#                 if t in banned:
#                     continue
#                 new_seq = seqs[beam_i] + [t]
#                 done = t in [EOS, PAD] or len(new_seq) >= max_seq_len
#                 candidates.append({
#                     "beam_src": beam_i,
#                     "seq": new_seq,
#                     "log_prob": log_probs[beam_i] + l,
#                     "done": done
#                 })
#                 count += 1

#         if not candidates:
#             break

#         candidates.sort(key=lambda x: score_fn(x["seq"], x["log_prob"], start), reverse=True)

#         kept, src = [], []
#         for c in candidates:
#             if c["done"]:
#                 completed.append(c)
#             elif len(kept) < beam_size:
#                 kept.append(c)
#                 src.append(c["beam_src"])

#         if not kept:
#             break

#         if len(completed) >= beam_size:
#             best_done  = max(completed, key=lambda x: score_fn(x["seq"], x["log_prob"], start))
#             best_alive = max(kept,      key=lambda x: score_fn(x["seq"], x["log_prob"], start))
#             if score_fn(best_done["seq"], best_done["log_prob"], start) >= \
#                score_fn(best_alive["seq"], best_alive["log_prob"], start):
#                 if early_stop:
#                     break
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= patience:
#                         break
#             else:
#                 patience_counter = 0

#         if src != list(range(len(src))):
#             src_t = torch.tensor(src, dtype=torch.long, device=device)
#             for k_buf, v_buf in kv_buffers:
#                 k_tmp = k_buf[src_t].clone()
#                 v_tmp = v_buf[src_t].clone()
#                 k_buf[:len(src)].copy_(k_tmp)
#                 v_buf[:len(src)].copy_(v_tmp)

#         unique_sets = [unique_sets[src[i]] | {c["seq"][-1]} for i, c in enumerate(kept)]
#         seqs        = [c["seq"] for c in kept]
#         log_probs   = [c["log_prob"] for c in kept]
#         dones       = [False] * len(kept)

#     pool = completed if completed else [{"seq": seqs[i], "log_prob": log_probs[i]} for i in range(len(seqs))]
#     return max(pool, key=lambda x: score_fn(x["seq"], x["log_prob"], start))

# # ================= CORE PAGED (kv cache + index_select reorder) =================
# def forward_paged_init(model, input_ids, beam_size, max_new_tokens):
#     prompt_len = len(input_ids)
#     max_total  = prompt_len + max_new_tokens

#     kv_buffers = []
#     for block in model.decoder_blocks:
#         k_buf = torch.zeros(beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device)
#         v_buf = torch.zeros(beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device)
#         kv_buffers.append((k_buf, v_buf))

#     prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

#     with torch.inference_mode():
#         first_logits, present_cache = model.prefill(prompt_tensor, kv_cache=None)
#         for i, (k, v) in enumerate(present_cache):
#             kv_buffers[i][0][:, :, :prompt_len, :].copy_(k.expand(beam_size, -1, -1, -1))
#             kv_buffers[i][1][:, :, :prompt_len, :].copy_(v.expand(beam_size, -1, -1, -1))

#     return first_logits[0], kv_buffers, prompt_len

# def beam_core_paged(model, input_ids, start,
#                     score_fn, penalty_fn,
#                     max_new_tokens, beam_size,
#                     no_repeat_ngram, penalty,
#                     early_stop=True, patience=10):
#     first_logits, kv_buffers, cache_len = forward_paged_init(
#         model, input_ids, beam_size, max_new_tokens
#     )

#     first_logits = penalty_fn(first_logits, input_ids, penalty)
#     first_lp     = torch.clamp(torch.log_softmax(first_logits, -1), -1e9, 0.0)
#     topk_lp, topk_tok = torch.topk(first_lp, beam_size)

#     seqs        = [input_ids + [int(t)] for t in topk_tok.tolist()]
#     log_probs   = topk_lp.tolist()
#     dones       = [int(t) in [EOS, PAD] for t in topk_tok.tolist()]
#     unique_sets = [set(input_ids) | {int(t)} for t in topk_tok.tolist()]
#     completed   = []
#     K           = beam_size * 3
#     patience_counter = 0

#     for _ in range(max_new_tokens - 1):
#         if all(dones):
#             break

#         last = torch.tensor([[seqs[i][-1]] for i in range(len(seqs))], dtype=torch.long, device=device)
#         logits_batch = forward_huong_2_step(model, last, kv_buffers, cache_len)
#         cache_len += 1

#         n_beams = len(seqs)
#         if penalty != 1.0:
#             pen_mask = torch.ones(n_beams, logits_batch.shape[-1], device=device)
#             for i, uid in enumerate(unique_sets):
#                 if dones[i]:
#                     continue
#                 idx = torch.tensor(list(uid), dtype=torch.long, device=device)
#                 pen_mask[i, idx] = penalty
#             neg = logits_batch < 0
#             logits_batch = torch.where(neg, logits_batch * pen_mask, logits_batch / pen_mask)

#         lp_batch = torch.clamp(torch.log_softmax(logits_batch, -1), -1e9, 0.0)
#         topk_lp_b, topk_tok_b = torch.topk(lp_batch, K, dim=-1)
#         topk_lp_b  = topk_lp_b.tolist()
#         topk_tok_b = topk_tok_b.tolist()

#         candidates = []
#         for beam_i in range(n_beams):
#             if dones[beam_i]:
#                 continue
#             banned = get_banned_tokens(seqs[beam_i], no_repeat_ngram)
#             count = 0
#             for l, t in zip(topk_lp_b[beam_i], topk_tok_b[beam_i]):
#                 if count >= beam_size:
#                     break
#                 if t in banned:
#                     continue
#                 new_seq = seqs[beam_i] + [t]
#                 done = t in [EOS, PAD] or len(new_seq) >= max_seq_len
#                 candidates.append({
#                     "beam_src": beam_i,
#                     "seq":      new_seq,
#                     "log_prob": log_probs[beam_i] + l,
#                     "done":     done,
#                 })
#                 count += 1

#         if not candidates:
#             break

#         candidates.sort(key=lambda x: score_fn(x["seq"], x["log_prob"], start), reverse=True)

#         kept, src = [], []
#         for c in candidates:
#             if c["done"]:
#                 completed.append(c)
#             elif len(kept) < beam_size:
#                 kept.append(c)
#                 src.append(c["beam_src"])

#         if not kept:
#             break

#         if len(completed) >= beam_size:
#             best_done  = max(completed, key=lambda x: score_fn(x["seq"], x["log_prob"], start))
#             best_alive = max(kept,      key=lambda x: score_fn(x["seq"], x["log_prob"], start))
#             if score_fn(best_done["seq"], best_done["log_prob"], start) >= \
#                score_fn(best_alive["seq"], best_alive["log_prob"], start):
#                 if early_stop:
#                     break
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= patience:
#                         break
#             else:
#                 patience_counter = 0

#         if src != list(range(len(src))):
#             src_t = torch.tensor(src, dtype=torch.long, device=device)
#             for k_buf, v_buf in kv_buffers:
#                 k_buf[:len(src), :, :cache_len, :] = k_buf[src_t, :, :cache_len, :]
#                 v_buf[:len(src), :, :cache_len, :] = v_buf[src_t, :, :cache_len, :]

#         unique_sets = [unique_sets[src[i]] | {c["seq"][-1]} for i, c in enumerate(kept)]
#         seqs        = [c["seq"]      for c in kept]
#         log_probs   = [c["log_prob"] for c in kept]
#         dones       = [False]        * len(kept)

#     pool = completed if completed else [{"seq": seqs[i], "log_prob": log_probs[i]} for i in range(len(seqs))]
#     return max(pool, key=lambda x: score_fn(x["seq"], x["log_prob"], start))

# # ================= 8 FUNCTIONS =================
# def generate_1(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_1(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

# def generate_3(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_1(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

# def generate_1_non_early_stop(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_2(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# def generate_3_non_early_stop(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_2(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# def generate_5(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_2(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

# def generate_5_non_early_stop(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_2(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# def generate_7(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_2(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

# def generate_7_non_early_stop(model, text):
#     ids, start = build_input(text)
#     return beam_core_huong_2(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# # ================= PAGED FUNCTIONS =================
# def _paged(model, text, score_fn, pen_fn, max_tok, bsz, ngram, pen,
#            early_stop=True, patience=10):
#     ids, start = build_input(text)
#     return beam_core_paged(
#         model, ids, start, score_fn, pen_fn,
#         max_tok, bsz, ngram, pen,
#         early_stop=early_stop, patience=patience,
#     )

# def generate_1_paged(model, text):
#     return _paged(model, text, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

# def generate_1_non_early_stop_paged(model, text):
#     return _paged(model, text, score_huong_1, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# def generate_3_paged(model, text):
#     return _paged(model, text, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

# def generate_3_non_early_stop_paged(model, text):
#     return _paged(model, text, score_huong_2, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# def generate_5_paged(model, text):
#     return _paged(model, text, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

# def generate_5_non_early_stop_paged(model, text):
#     return _paged(model, text, score_huong_1, apply_penalty,
#                   200, 5, 3, 1.2, early_stop=False, patience=30)

# def generate_7_paged(model, text):
#     return _paged(model, text, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

# def generate_7_non_early_stop_paged(model, text):
#     return _paged(model, text, score_huong_2, apply_penalty,
#                   200, 5, 3, 1.2, early_stop=False, patience=30)

# generators = [
#     # ── no-cache (huong_1) ──────────────────────────────────────────────
#     ("gen1",                      generate_1,                      score_huong_1),
#     ("gen1_non_early_stop",       generate_1_non_early_stop,       score_huong_1),  # NOTE: beam_core_huong_1 ignores early_stop; non_early_stop variant uses beam_core_huong_2
#     ("gen3",                      generate_3,                      score_huong_2),
#     ("gen3_non_early_stop",       generate_3_non_early_stop,       score_huong_2),
#     # ── kv-cache (huong_2) ──────────────────────────────────────────────
#     ("gen5",                      generate_5,                      score_huong_1),
#     ("gen5_non_early_stop",       generate_5_non_early_stop,       score_huong_1),
#     ("gen7",                      generate_7,                      score_huong_2),
#     ("gen7_non_early_stop",       generate_7_non_early_stop,       score_huong_2),
#     # ── paged ───────────────────────────────────────────────────────────
#     ("gen1_paged",                generate_1_paged,                score_huong_1),
#     ("gen1_non_early_stop_paged", generate_1_non_early_stop_paged, score_huong_1),
#     ("gen3_paged",                generate_3_paged,                score_huong_2),
#     ("gen3_non_early_stop_paged", generate_3_non_early_stop_paged, score_huong_2),
#     ("gen5_paged",                generate_5_paged,                score_huong_1),
#     ("gen5_non_early_stop_paged", generate_5_non_early_stop_paged, score_huong_1),
#     ("gen7_paged",                generate_7_paged,                score_huong_2),
#     ("gen7_non_early_stop_paged", generate_7_non_early_stop_paged, score_huong_2),
# ]

# model = models["sft1"]["model"]

# all_results = []

# for text in test_cases:
#     print("\n" + "="*60)
#     print("INPUT:", text)

#     case_results = []

#     for name, fn, score_fn in generators:
#         t0 = time.time()

#         best = fn(model, text)

#         dt = time.time() - t0

#         ids, start = build_input(text)
#         output_tokens = best["seq"][start:]

#         while output_tokens and output_tokens[-1] in [EOS, PAD]:
#             output_tokens.pop()

#         out_text = tokenizer.decode(output_tokens)

#         tok = len(output_tokens)
#         tps = tok / dt if dt > 0 else 0

#         score = score_fn(best["seq"], best["log_prob"], start)

#         print(f"\n{name}")
#         print(f"→ {out_text}")
#         print(f"time={dt:.3f}s | tok={tok} | tps={tps:.2f} | score={score:.4f}")

#         case_results.append({
#             "name": name,
#             "output": out_text,
#             "time": dt,
#             "tokens": tok,
#             "tps": tps,
#             "score": score
#         })

#     all_results.append({
#         "input": text,
#         "runs": case_results
#     })
        
# # ================= AGGREGATE =================
# speed_stats  = {}
# length_stats = {}

# for case in all_results:
#     for r in case["runs"]:
#         name = r["name"]
#         speed_stats.setdefault(name,  []).append(r["tps"])
#         length_stats.setdefault(name, []).append(r["tokens"])

# avg_speed  = {k: sum(v)/len(v) for k, v in speed_stats.items()}
# avg_length = {k: sum(v)/len(v) for k, v in length_stats.items()}

# speed_ranking  = sorted(avg_speed.items(),  key=lambda x: x[1], reverse=True)
# length_ranking = sorted(avg_length.items(), key=lambda x: x[1], reverse=True)

# # ── Taxonomy helpers ──────────────────────────────────────────────────────────
# def is_non_early_stop(name): return "non_early_stop" in name
# def is_paged(name):          return "paged" in name
# def is_early_stop(name):     return not is_non_early_stop(name)

# GROUP_NORMAL               = "normal"
# GROUP_NON_EARLY_STOP       = "non_early_stop"
# GROUP_PAGED                = "paged"
# GROUP_NON_EARLY_STOP_PAGED = "non_early_stop_paged"
# ALL_GROUPS = [GROUP_NORMAL, GROUP_NON_EARLY_STOP, GROUP_PAGED, GROUP_NON_EARLY_STOP_PAGED]

# def get_group(name):
#     p, n = is_paged(name), is_non_early_stop(name)
#     if   not p and not n: return GROUP_NORMAL
#     elif not p and     n: return GROUP_NON_EARLY_STOP
#     elif     p and not n: return GROUP_PAGED
#     else:                 return GROUP_NON_EARLY_STOP_PAGED

# def get_cache_fn(name):
#     for i in [5,6,7,8]:
#         if f"gen{i}" in name: return "kv_cache"
#     return "no_cache"
# def get_score_fn(name):
#     return "score2" if any(f"gen{i}" in name for i in [3,4,7,8]) else "score1"

# def group_avg(names_list, metric_dict):
#     vals = [metric_dict[n] for n in names_list if n in metric_dict]
#     return sum(vals) / len(vals) if vals else 0.0

# all_names   = list(avg_speed.keys())
# group_names = {g: [n for n in all_names if get_group(n) == g] for g in ALL_GROUPS}

# sub_no_cache       = [n for n in all_names if get_cache_fn(n)   == "no_cache"]
# sub_kv_cache       = [n for n in all_names if get_cache_fn(n)   == "kv_cache"]
# sub_score1         = [n for n in all_names if get_score_fn(n)   == "score1"]
# sub_score2         = [n for n in all_names if get_score_fn(n)   == "score2"]
# sub_early_stop     = [n for n in all_names if is_early_stop(n)]
# sub_non_early_stop = [n for n in all_names if is_non_early_stop(n)]
# sub_paged          = [n for n in all_names if is_paged(n)]
# sub_non_paged      = [n for n in all_names if not is_paged(n)]

# # ================= PRINT RANK =================
# SEP = "=" * 60

# def print_ranking(title, items, metric_label):
#     # items is already sorted; enforce descending just in case
#     sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
#     print(f"\n{title}")
#     print(f"  {'#':<4} {'name':<35} {metric_label:>10}")
#     print(f"  {'-'*4} {'-'*35} {'-'*10}")
#     for i, (name, val) in enumerate(sorted_items, 1):
#         print(f"  {i:<4} {name:<35} {val:>10.2f}")

# def print_group_summary(title, metric_dict, metric_label):
#     # sort groups by metric descending
#     rows = sorted(
#         [(g, group_avg(group_names[g], metric_dict), len(group_names[g])) for g in ALL_GROUPS],
#         key=lambda x: x[1], reverse=True
#     )
#     print(f"\n{title}")
#     print(f"  {'group':<28} {metric_label:>10}  n")
#     print(f"  {'-'*28} {'-'*10}  -")
#     for g, avg, n in rows:
#         print(f"  {g:<28} {avg:>10.2f}  ({n})")

# def print_sub_comparison(title, sub_pairs, metric_dict, metric_label):
#     # sort pairs by metric descending
#     rows = sorted(
#         [(label, group_avg(names, metric_dict), len(names)) for label, names in sub_pairs],
#         key=lambda x: x[1], reverse=True
#     )
#     print(f"\n{title}")
#     print(f"  {'dimension':<30} {metric_label:>10}  n")
#     print(f"  {'-'*30} {'-'*10}  -")
#     for label, avg, n in rows:
#         print(f"  {label:<30} {avg:>10.2f}  {n}")

# print("\n" + SEP)
# print("📊  BENCHMARK SUMMARY")
# print(SEP)

# # 1. Global rankings (speed + length only)
# print_ranking("🏁  SPEED RANKING — ALL (tok/s)",   speed_ranking,  "tok/s")
# print_ranking("📏  LENGTH RANKING — ALL (tokens)", length_ranking, "tokens")

# # 2. Group averages
# print("\n" + SEP)
# print("📋  GROUP AVERAGES")
# print(SEP)
# print_group_summary("⚡  Avg Speed by group (tok/s)",   avg_speed,  "tok/s")
# print_group_summary("📐  Avg Length by group (tokens)", avg_length, "tokens")

# # 3. Sub-dimension comparisons
# print("\n" + SEP)
# print("🔬  SUB-DIMENSION COMPARISONS")
# print(SEP)

# pairs_es = [("early_stop",     sub_early_stop),     ("non_early_stop", sub_non_early_stop)]
# pairs_pg = [("non_paged",      sub_non_paged),       ("paged",          sub_paged)]
# pairs_kv = [("no_cache gen1-4", sub_no_cache),       ("kv_cache gen5-8", sub_kv_cache)]
# pairs_sf = [("score_fn1 gen1,2,5,6", sub_score1),   ("score_fn2 gen3,4,7,8", sub_score2)]

# print_sub_comparison("⚡  Speed: early_stop vs non_early_stop",  pairs_es, avg_speed,  "tok/s")
# print_sub_comparison("📐  Length: early_stop vs non_early_stop", pairs_es, avg_length, "tokens")

# print_sub_comparison("⚡  Speed: paged vs non_paged",            pairs_pg, avg_speed,  "tok/s")
# print_sub_comparison("📐  Length: paged vs non_paged",           pairs_pg, avg_length, "tokens")

# print_sub_comparison("⚡  Speed: no_cache vs kv_cache",          pairs_kv, avg_speed,  "tok/s")
# print_sub_comparison("📐  Length: no_cache vs kv_cache",         pairs_kv, avg_length, "tokens")

# print_sub_comparison("⚡  Speed: score_fn1 vs score_fn2",        pairs_sf, avg_speed,  "tok/s")
# print_sub_comparison("📐  Length: score_fn1 vs score_fn2",       pairs_sf, avg_length, "tokens")

# # ================= SAVE MARKDOWN =================
# MD = "/Users/thongbui.nd/Documents/Thong Bui/dev_2/src/test/benchmark_results.md"

# def md_ranking_table(items, col):
#     sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
#     lines = [f"| # | name | {col} |", "|---|------|------|"]
#     for i, (name, val) in enumerate(sorted_items, 1):
#         lines.append(f"| {i} | `{name}` | {val:.2f} |")
#     return "\n".join(lines)

# def md_group_summary_table(metric_dict, col):
#     rows = sorted(
#         [(g, group_avg(group_names[g], metric_dict), len(group_names[g])) for g in ALL_GROUPS],
#         key=lambda x: x[1], reverse=True
#     )
#     lines = [f"| group | {col} | n |", "|-------|------|---|"]
#     for g, avg, n in rows:
#         lines.append(f"| `{g}` | {avg:.2f} | {n} |")
#     return "\n".join(lines)

# def md_sub_comparison_table(sub_pairs, metric_dict, col):
#     rows = sorted(
#         [(label, group_avg(names, metric_dict), len(names)) for label, names in sub_pairs],
#         key=lambda x: x[1], reverse=True
#     )
#     lines = [f"| dimension | {col} | n |", "|-----------|------|---|"]
#     for label, avg, n in rows:
#         lines.append(f"| {label} | {avg:.2f} | {n} |")
#     return "\n".join(lines)

# with open(MD, "w", encoding="utf-8") as f:

#     # ── Raw results ──────────────────────────────────────────────────────────
#     f.write("# 📝 Raw Results\n\n")
#     for case in all_results:
#         f.write(f"## INPUT: {case['input']}\n\n")
#         f.write("| name | output | time (s) | tokens | tok/s |\n")
#         f.write("|------|--------|----------|--------|-------|\n")
#         for r in case["runs"]:
#             out_escaped = r["output"].replace("|", "\\|")
#             f.write(f"| `{r['name']}` | {out_escaped} | {r['time']:.3f} | {r['tokens']} | {r['tps']:.2f} |\n")
#         f.write("\n---\n\n")

#     # ── Global rankings ──────────────────────────────────────────────────────
#     f.write("# 🏁 Global Rankings\n\n")
#     f.write("## Speed — all (tok/s)\n\n")
#     f.write(md_ranking_table(speed_ranking, "tok/s"))
#     f.write("\n\n## Length — all (tokens)\n\n")
#     f.write(md_ranking_table(length_ranking, "tokens"))
#     f.write("\n\n")

#     # ── Group averages ───────────────────────────────────────────────────────
#     f.write("# 📋 Group Averages\n\n")
#     f.write("## Avg Speed by group (tok/s)\n\n")
#     f.write(md_group_summary_table(avg_speed, "tok/s"))
#     f.write("\n\n## Avg Length by group (tokens)\n\n")
#     f.write(md_group_summary_table(avg_length, "tokens"))
#     f.write("\n\n")

#     # ── Sub-dimension comparisons ────────────────────────────────────────────
#     f.write("# 🔬 Sub-Dimension Comparisons\n\n")

#     for section_title, pairs in [
#         ("early_stop vs non_early_stop", pairs_es),
#         ("paged vs non_paged",           pairs_pg),
#         ("no_cache vs kv_cache",         pairs_kv),
#         ("score_fn1 vs score_fn2",       pairs_sf)
#     ]:
#         f.write(f"## {section_title}\n\n")
#         f.write("### Speed (tok/s)\n\n")
#         f.write(md_sub_comparison_table(pairs, avg_speed,  "tok/s"))
#         f.write("\n\n### Length (tokens)\n\n")
#         f.write(md_sub_comparison_table(pairs, avg_length, "tokens"))
#         f.write("\n\n")

# print(f"\n✅  Đã lưu kết quả vào: {MD}")
































import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
import torch
import json
import time
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
from tokenizers import Tokenizer
from src.model import TransformerModel
from src.paged_attention import PagedKVPool, BlockTable

config_dir = project_root / "config"
base_config_file = config_dir / "base.json"
model_config_file = config_dir / f"35M.json"

sft1_file = project_root / "model" / "sft1_35M.pt"
data_dir = project_root / "data"

with open(base_config_file, 'r') as f:
    config = json.load(f)
with open(model_config_file, 'r') as f:
    config.update(json.load(f))
vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️  Sử dụng device: {device}")

tokenizer_file = data_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()

USER = vocab["<|user|>"]
SAI = vocab["<|s.a.i|>"]
BOS = vocab["[BOS]"]
EOS = vocab["[EOS]"]
PAD = vocab["[PAD]"]

def load_model(model_file):
    model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    return model

models = {
    "sft1": {"model": load_model(sft1_file)}
}
print(f"✅ Đã load {len(models)} models: {', '.join(models.keys())}")
total_params = 0
for param_name, param in models["sft1"]["model"].named_parameters():
    total_params += param.numel()
        
print(f"👉 Tổng số tham số của model: {total_params:,}")

def pad_sequence(sequence, max_len, padding_value=0):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [padding_value] * (max_len - len(sequence))

def generate_response_prev(model, user_input, max_new_tokens=50, beam_size=5):
    model.eval()
    prompt = " Input: " + user_input
    prompt_ids = tokenizer.encode(prompt).ids
    input = [BOS] + [USER] + prompt_ids + [SAI]
    output_start_idx = len(input)

    beams = [{"seq": list(input), "log_prob": 0.0, "done": False}]
    completed_beams = []

    def normalized_score(beam):
        out_len = max(len(beam["seq"]) - output_start_idx, 1)
        return beam["log_prob"] / (out_len ** 1.0)

    for step in range(max_new_tokens):
        active_beams = [b for b in beams if not b["done"]]
        if not active_beams:
            break

        batch_inputs = [pad_sequence(b["seq"], max_seq_len, padding_value=PAD) for b in active_beams]
        batch_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)

        attention_mask = (batch_tensor != PAD).long()

        with torch.inference_mode():
            logits_batch = model(batch_tensor, attention_mask=attention_mask)

        all_candidates = []
        for b_idx, beam in enumerate(active_beams):
            cur_pos = len(beam["seq"]) - 1
            logits = logits_batch[b_idx, cur_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.clamp(log_probs, -1e9, 0.0)

            topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

            for log_p, token in zip(topk_log_probs.cpu().numpy(), topk_tokens.cpu().numpy()):
                token = int(token)
                new_seq = beam["seq"] + [token]
                new_log_prob = beam["log_prob"] + float(log_p)
                done = token in [EOS, PAD] or len(new_seq) >= max_seq_len

                all_candidates.append({
                    "seq": new_seq,
                    "log_prob": new_log_prob,
                    "done": done
                })

        all_candidates.sort(key=normalized_score, reverse=True)
        kept = all_candidates[:beam_size]

        beams = []
        for cand in kept:
            if cand["done"]:
                completed_beams.append(cand)
            else:
                beams.append(cand)

        if len(completed_beams) >= beam_size:
            break

    final_pool = completed_beams if completed_beams else beams
    best_beam = max(final_pool, key=normalized_score)

    output_tokens = best_beam["seq"][output_start_idx:]
    while output_tokens and output_tokens[-1] in [EOS, PAD]:
        output_tokens.pop()

    return tokenizer.decode(output_tokens)

test_cases = [
    "formula 1 là gì",
    "tôi muốn tìm hiểu về giải đua công thức 1",
    "formula one là giải đua gì",
    "oracle redbull racing",
    "redbull racing",
    "mercedes-amg petronat formula one team",
    "cách làm bánh mì việt nam",
    "cho tôi thêm thông tin về đội đua f1 ferrari",
    "max verstappen là ai",
    "ẩm thực việt nam",
    "cách nấu phở",
    "cách làm pizza",
    "hướng dẫn tôi làm một ly sinh tố dâu tây thật mát lạnh trong ngày hè nóng bức",
    "tôi muốn làm một lý sinh tố bơ",
    "hướng dẫn cho tôi cách nấu cháo gà",
    "làm hamburger tại nhà",
    "manchester united",
    "sir alex Ferguson là ai",
    "sir alex đã vô địch ngoại hạng anh mấy lần",
    "kim dahyun là ai",
    "twice gồm mấy thành viên",
    "huyết áp trung bình (MAP) là gì",
    "kỹ thuật xoáy xuống trong bóng bàn",
    "thông tin về bts",
    "bts có mấy thành viên",
    "sơn tùng m-tp là ai"
]

# ================= COMMON =================
def build_input(user_input):
    prompt = " Input: " + user_input
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = [BOS] + [USER] + prompt_ids + [SAI]
    return input_ids, len(input_ids)

def get_banned_tokens(seq, n):
    banned = set()
    if n > 0 and len(seq) >= n:
        prefix = tuple(seq[-(n - 1):])
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
    return banned

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

def score_huong_1(seq, log_prob, start):
    out_len = max(len(seq) - start, 1)
    return log_prob / (out_len ** 1.0)

def score_huong_2(seq, log_prob, start):
    out_len = max(len(seq) - start, 1)
    lp = ((5 + out_len) / 6) ** 0.6
    return log_prob / lp

# ================= FORWARD =================
def forward_huong_1(model, beams):
    max_len = max(len(b["seq"]) for b in beams)
    batch_inputs = [pad_sequence(b["seq"], max_len, PAD) for b in beams]
    batch_tensor = torch.as_tensor(batch_inputs, dtype=torch.long, device=device)
    attention_mask = (batch_tensor != PAD).long()

    with torch.inference_mode():
        logits_batch = model(batch_tensor, attention_mask=attention_mask)

    out = []
    for i, b in enumerate(beams):
        cur_pos = len(b["seq"]) - 1
        out.append(logits_batch[i, cur_pos, :])
    return out

def forward_huong_2_init(model, input_ids, max_beam_size, max_new_tokens):
    prompt_len = len(input_ids)
    max_total = prompt_len + max_new_tokens

    kv_buffers = []
    for block in model.blocks:
        k_buf = torch.zeros(max_beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device)
        v_buf = torch.zeros(max_beam_size, block.mha.num_heads, max_total, block.mha.d_k, device=device)
        kv_buffers.append((k_buf, v_buf))

    prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        first_logits, present_cache = model.prefill(prompt_tensor, kv_cache=None)
        for i, (k, v) in enumerate(present_cache):
            kv_buffers[i][0][:, :, :prompt_len, :].copy_(k.expand(max_beam_size, -1, -1, -1))
            kv_buffers[i][1][:, :, :prompt_len, :].copy_(v.expand(max_beam_size, -1, -1, -1))

    return first_logits[0], kv_buffers, prompt_len

def forward_huong_2_step(model, last_tokens_full, kv_buffers, cache_len):
    with torch.inference_mode():
        x = model.embed(last_tokens_full)

        for l, block in enumerate(model.blocks):
            x = block.forward_with_cache(x, kv_buffers[l], cache_len)

        logits = model.lm_head(model.norm(x))[:, 0, :]

    return logits

# ================= CORE 1 (no cache) =================
def beam_core_huong_1(model, input_ids, start,
                     score_fn, penalty_fn,
                     max_new_tokens, beam_size,
                     no_repeat_ngram, penalty):

    beams = [{"seq": list(input_ids), "log_prob": 0.0, "done": False}]
    completed = []

    for _ in range(max_new_tokens):
        active = [b for b in beams if not b["done"]]
        if not active:
            break

        logits_list = forward_huong_1(model, active)

        candidates = []
        for beam, logits in zip(active, logits_list):
            logits = penalty_fn(logits, beam["seq"], penalty)

            lp = torch.clamp(torch.log_softmax(logits, -1), -1e9, 0.0)
            banned = get_banned_tokens(beam["seq"], no_repeat_ngram)

            topk_lp, topk_tok = torch.topk(lp, beam_size * 3)

            count = 0
            for l, t in zip(topk_lp.tolist(), topk_tok.tolist()):
                if count >= beam_size:
                    break
                if t in banned:
                    continue

                new_seq = beam["seq"] + [t]
                done = t in [EOS, PAD] or len(new_seq) >= max_seq_len

                candidates.append({
                    "seq": new_seq,
                    "log_prob": beam["log_prob"] + l,
                    "done": done
                })
                count += 1

        candidates.sort(key=lambda x: score_fn(x["seq"], x["log_prob"], start), reverse=True)

        beams = []
        for c in candidates[:beam_size]:
            if c["done"]:
                completed.append(c)
            else:
                beams.append(c)

        if len(completed) >= beam_size:
            break

    pool = completed if completed else beams
    return max(pool, key=lambda x: score_fn(x["seq"], x["log_prob"], start))

# ================= CORE 2 (kv cache) =================
def beam_core_huong_2(model, input_ids, start,
                     score_fn, penalty_fn,
                     max_new_tokens, beam_size,
                     no_repeat_ngram, penalty,
                     early_stop=True, patience=10):

    first_logits, kv_buffers, cache_len = forward_huong_2_init(
        model, input_ids, beam_size, max_new_tokens
    )

    first_logits = penalty_fn(first_logits, input_ids, penalty)
    first_lp = torch.clamp(torch.log_softmax(first_logits, -1), -1e9, 0.0)
    topk_lp, topk_tok = torch.topk(first_lp, beam_size)

    seqs        = [input_ids + [int(t)] for t in topk_tok.tolist()]
    log_probs   = topk_lp.tolist()
    dones       = [int(t) in [EOS, PAD] for t in topk_tok.tolist()]
    unique_sets = [set(input_ids) | {int(t)} for t in topk_tok.tolist()]
    completed   = []
    K           = beam_size * 3
    patience_counter = 0

    for _ in range(max_new_tokens - 1):
        if all(dones):
            break

        last = torch.tensor([[seqs[i][-1]] for i in range(len(seqs))], dtype=torch.long, device=device)
        logits_batch = forward_huong_2_step(model, last, kv_buffers, cache_len)
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
        topk_lp_b  = topk_lp_b.tolist()
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
            best_done  = max(completed, key=lambda x: score_fn(x["seq"], x["log_prob"], start))
            best_alive = max(kept,      key=lambda x: score_fn(x["seq"], x["log_prob"], start))
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
        seqs        = [c["seq"] for c in kept]
        log_probs   = [c["log_prob"] for c in kept]
        dones       = [False] * len(kept)

    pool = completed if completed else [{"seq": seqs[i], "log_prob": log_probs[i]} for i in range(len(seqs))]
    return max(pool, key=lambda x: score_fn(x["seq"], x["log_prob"], start))

# ================= CORE PAGED (PagedKVPool) =================
def forward_paged_init(model, input_ids, beam_size, max_new_tokens):
    """
    Prefill prompt vào PagedKVPool (batch=1).
    Trả về: first_logits (vocab,), danh sách pool+block_table cho mỗi beam, prompt_len.

    Vì PagedKVPool hỗ trợ batch=1, mỗi beam có pool + block_table riêng.
    Sau prefill, beam 0 là gốc — các beam 1..K-1 copy block_table từ beam 0.
    """
    prompt_len = len(input_ids)
    prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Tạo pool đủ lớn cho tất cả beam
    # num_blocks: mỗi beam cần ceil((prompt_len+max_new_tokens)/block_size) block
    block_size = 16
    blocks_per_seq = math.ceil((prompt_len + max_new_tokens) / block_size) + 1
    total_blocks = blocks_per_seq * beam_size + 64   # buffer dư

    pool = model.make_pool(device, num_blocks=total_blocks, block_size=block_size,
                           dtype=next(model.parameters()).dtype)

    # Prefill beam 0
    block_tables = [BlockTable() for _ in range(beam_size)]
    with torch.inference_mode():
        first_logits = model.prefill_paged(prompt_tensor, pool, block_tables[0])

    # Copy physical blocks của beam 0 sang beam 1..K-1
    src_pids = block_tables[0].physical_ids()
    for b in range(1, beam_size):
        for pid in src_pids:
            new_pid = pool.allocate_block()
            block_tables[b].append_block(new_pid)
        pool.copy_blocks(src_pids, block_tables[b].physical_ids())

    return first_logits[0], pool, block_tables, prompt_len


def forward_paged_step(model, last_tokens, pool, block_tables, seq_len):
    """
    Decode một bước cho tất cả beam song song (loop vì batch=1).
    last_tokens: list[int] độ dài n_beams.
    Trả về logits tensor (n_beams, vocab_size).
    """
    logits_list = []
    with torch.inference_mode():
        for b, tok in enumerate(last_tokens):
            tok_t = torch.tensor([tok], dtype=torch.long, device=device)
            logits = model.decode_paged(tok_t, pool, block_tables[b], seq_len)
            logits_list.append(logits)   # (1, vocab_size)
    return torch.cat(logits_list, dim=0)  # (n_beams, vocab_size)


def beam_core_paged(model, input_ids, start,
                    score_fn, penalty_fn,
                    max_new_tokens, beam_size,
                    no_repeat_ngram, penalty,
                    early_stop=True, patience=10):
    first_logits, pool, block_tables, cache_len = forward_paged_init(
        model, input_ids, beam_size, max_new_tokens
    )

    first_logits = penalty_fn(first_logits, input_ids, penalty)
    first_lp     = torch.clamp(torch.log_softmax(first_logits, -1), -1e9, 0.0)
    topk_lp, topk_tok = torch.topk(first_lp, beam_size)

    seqs        = [input_ids + [int(t)] for t in topk_tok.tolist()]
    log_probs   = topk_lp.tolist()
    dones       = [int(t) in [EOS, PAD] for t in topk_tok.tolist()]
    unique_sets = [set(input_ids) | {int(t)} for t in topk_tok.tolist()]
    completed   = []
    K            = beam_size * 3
    patience_counter = 0

    for _ in range(max_new_tokens - 1):
        if all(dones):
            break

        active_last = [seqs[i][-1] for i in range(len(seqs))]
        logits_batch = forward_paged_step(model, active_last, pool, block_tables, cache_len)
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
        topk_lp_b  = topk_lp_b.tolist()
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
                    "seq":      new_seq,
                    "log_prob": log_probs[beam_i] + l,
                    "done":     done,
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
            best_done  = max(completed, key=lambda x: score_fn(x["seq"], x["log_prob"], start))
            best_alive = max(kept,      key=lambda x: score_fn(x["seq"], x["log_prob"], start))
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

        # Beam reorder: copy physical blocks nếu src thay đổi
        if src != list(range(len(src))):
            # Tạo block_tables mới tương ứng với src
            new_block_tables = []
            for new_b, old_b in enumerate(src):
                if new_b == old_b:
                    new_block_tables.append(block_tables[old_b])
                else:
                    # Cấp phát block mới và copy từ src
                    src_pids = block_tables[old_b].physical_ids()
                    new_bt = BlockTable()
                    for _ in src_pids:
                        new_pid = pool.allocate_block()
                        new_bt.append_block(new_pid)
                    pool.copy_blocks(src_pids, new_bt.physical_ids())
                    new_block_tables.append(new_bt)
            # Giải phóng block_table cũ không còn dùng
            old_indices_kept = set(src)
            for b in range(len(block_tables)):
                if b not in old_indices_kept:
                    pool.free_sequence(block_tables[b])
            block_tables = new_block_tables

        unique_sets = [unique_sets[src[i]] | {c["seq"][-1]} for i, c in enumerate(kept)]
        seqs        = [c["seq"]      for c in kept]
        log_probs   = [c["log_prob"] for c in kept]
        dones       = [False]        * len(kept)

    # Giải phóng pool
    for bt in block_tables:
        pool.free_sequence(bt)

    pool_final = completed if completed else [{"seq": seqs[i], "log_prob": log_probs[i]} for i in range(len(seqs))]
    return max(pool_final, key=lambda x: score_fn(x["seq"], x["log_prob"], start))

# ================= 8 FUNCTIONS =================
def generate_1(model, text):
    ids, start = build_input(text)
    return beam_core_huong_1(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

def generate_3(model, text):
    ids, start = build_input(text)
    return beam_core_huong_1(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

def generate_1_non_early_stop(model, text):
    ids, start = build_input(text)
    return beam_core_huong_2(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

def generate_3_non_early_stop(model, text):
    ids, start = build_input(text)
    return beam_core_huong_2(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

def generate_5(model, text):
    ids, start = build_input(text)
    return beam_core_huong_2(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

def generate_5_non_early_stop(model, text):
    ids, start = build_input(text)
    return beam_core_huong_2(model, ids, start, score_huong_1, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

def generate_7(model, text):
    ids, start = build_input(text)
    return beam_core_huong_2(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

def generate_7_non_early_stop(model, text):
    ids, start = build_input(text)
    return beam_core_huong_2(model, ids, start, score_huong_2, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

# ================= PAGED FUNCTIONS =================
def _paged(model, text, score_fn, pen_fn, max_tok, bsz, ngram, pen,
           early_stop=True, patience=10):
    ids, start = build_input(text)
    return beam_core_paged(
        model, ids, start, score_fn, pen_fn,
        max_tok, bsz, ngram, pen,
        early_stop=early_stop, patience=patience,
    )

def generate_1_paged(model, text):
    return _paged(model, text, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

def generate_1_non_early_stop_paged(model, text):
    return _paged(model, text, score_huong_1, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

def generate_3_paged(model, text):
    return _paged(model, text, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

def generate_3_non_early_stop_paged(model, text):
    return _paged(model, text, score_huong_2, apply_penalty, 200, 5, 3, 1.2, early_stop=False, patience=30)

def generate_5_paged(model, text):
    return _paged(model, text, score_huong_1, apply_penalty, 200, 5, 3, 1.2)

def generate_5_non_early_stop_paged(model, text):
    return _paged(model, text, score_huong_1, apply_penalty,
                  200, 5, 3, 1.2, early_stop=False, patience=30)

def generate_7_paged(model, text):
    return _paged(model, text, score_huong_2, apply_penalty, 200, 5, 3, 1.2)

def generate_7_non_early_stop_paged(model, text):
    return _paged(model, text, score_huong_2, apply_penalty,
                  200, 5, 3, 1.2, early_stop=False, patience=30)

generators = [
    # ── no-cache (huong_1) ──────────────────────────────────────────────
    ("gen1",                      generate_1,                      score_huong_1),
    ("gen1_non_early_stop",       generate_1_non_early_stop,       score_huong_1),  # NOTE: beam_core_huong_1 ignores early_stop; non_early_stop variant uses beam_core_huong_2
    ("gen3",                      generate_3,                      score_huong_2),
    ("gen3_non_early_stop",       generate_3_non_early_stop,       score_huong_2),
    # ── kv-cache (huong_2) ──────────────────────────────────────────────
    ("gen5",                      generate_5,                      score_huong_1),
    ("gen5_non_early_stop",       generate_5_non_early_stop,       score_huong_1),
    ("gen7",                      generate_7,                      score_huong_2),
    ("gen7_non_early_stop",       generate_7_non_early_stop,       score_huong_2),
    # ── paged ───────────────────────────────────────────────────────────
    ("gen1_paged",                generate_1_paged,                score_huong_1),
    ("gen1_non_early_stop_paged", generate_1_non_early_stop_paged, score_huong_1),
    ("gen3_paged",                generate_3_paged,                score_huong_2),
    ("gen3_non_early_stop_paged", generate_3_non_early_stop_paged, score_huong_2),
    ("gen5_paged",                generate_5_paged,                score_huong_1),
    ("gen5_non_early_stop_paged", generate_5_non_early_stop_paged, score_huong_1),
    ("gen7_paged",                generate_7_paged,                score_huong_2),
    ("gen7_non_early_stop_paged", generate_7_non_early_stop_paged, score_huong_2),
]

model = models["sft1"]["model"]

all_results = []

for text in test_cases:
    print("\n" + "="*60)
    print("INPUT:", text)

    case_results = []

    for name, fn, score_fn in generators:
        t0 = time.time()

        best = fn(model, text)

        dt = time.time() - t0

        ids, start = build_input(text)
        output_tokens = best["seq"][start:]

        while output_tokens and output_tokens[-1] in [EOS, PAD]:
            output_tokens.pop()

        out_text = tokenizer.decode(output_tokens)

        tok = len(output_tokens)
        tps = tok / dt if dt > 0 else 0

        score = score_fn(best["seq"], best["log_prob"], start)

        print(f"\n{name}")
        print(f"→ {out_text}")
        print(f"time={dt:.3f}s | tok={tok} | tps={tps:.2f} | score={score:.4f}")

        case_results.append({
            "name": name,
            "output": out_text,
            "time": dt,
            "tokens": tok,
            "tps": tps,
            "score": score
        })

    all_results.append({
        "input": text,
        "runs": case_results
    })
        
# ================= AGGREGATE =================
speed_stats  = {}
length_stats = {}

for case in all_results:
    for r in case["runs"]:
        name = r["name"]
        speed_stats.setdefault(name,  []).append(r["tps"])
        length_stats.setdefault(name, []).append(r["tokens"])

avg_speed  = {k: sum(v)/len(v) for k, v in speed_stats.items()}
avg_length = {k: sum(v)/len(v) for k, v in length_stats.items()}

speed_ranking  = sorted(avg_speed.items(),  key=lambda x: x[1], reverse=True)
length_ranking = sorted(avg_length.items(), key=lambda x: x[1], reverse=True)

# ── Taxonomy helpers ──────────────────────────────────────────────────────────
def is_non_early_stop(name): return "non_early_stop" in name
def is_paged(name):          return "paged" in name
def is_early_stop(name):     return not is_non_early_stop(name)

GROUP_NORMAL               = "normal"
GROUP_NON_EARLY_STOP       = "non_early_stop"
GROUP_PAGED                = "paged"
GROUP_NON_EARLY_STOP_PAGED = "non_early_stop_paged"
ALL_GROUPS = [GROUP_NORMAL, GROUP_NON_EARLY_STOP, GROUP_PAGED, GROUP_NON_EARLY_STOP_PAGED]

def get_group(name):
    p, n = is_paged(name), is_non_early_stop(name)
    if   not p and not n: return GROUP_NORMAL
    elif not p and     n: return GROUP_NON_EARLY_STOP
    elif     p and not n: return GROUP_PAGED
    else:                 return GROUP_NON_EARLY_STOP_PAGED

def get_cache_fn(name):
    for i in [5,6,7,8]:
        if f"gen{i}" in name: return "kv_cache"
    return "no_cache"
def get_score_fn(name):
    return "score2" if any(f"gen{i}" in name for i in [3,4,7,8]) else "score1"

def group_avg(names_list, metric_dict):
    vals = [metric_dict[n] for n in names_list if n in metric_dict]
    return sum(vals) / len(vals) if vals else 0.0

all_names   = list(avg_speed.keys())
group_names = {g: [n for n in all_names if get_group(n) == g] for g in ALL_GROUPS}

sub_no_cache       = [n for n in all_names if get_cache_fn(n)   == "no_cache"]
sub_kv_cache       = [n for n in all_names if get_cache_fn(n)   == "kv_cache"]
sub_score1         = [n for n in all_names if get_score_fn(n)   == "score1"]
sub_score2         = [n for n in all_names if get_score_fn(n)   == "score2"]
sub_early_stop     = [n for n in all_names if is_early_stop(n)]
sub_non_early_stop = [n for n in all_names if is_non_early_stop(n)]
sub_paged          = [n for n in all_names if is_paged(n)]
sub_non_paged      = [n for n in all_names if not is_paged(n)]

# ================= PRINT RANK =================
SEP = "=" * 60

def print_ranking(title, items, metric_label):
    # items is already sorted; enforce descending just in case
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    print(f"\n{title}")
    print(f"  {'#':<4} {'name':<35} {metric_label:>10}")
    print(f"  {'-'*4} {'-'*35} {'-'*10}")
    for i, (name, val) in enumerate(sorted_items, 1):
        print(f"  {i:<4} {name:<35} {val:>10.2f}")

def print_group_summary(title, metric_dict, metric_label):
    # sort groups by metric descending
    rows = sorted(
        [(g, group_avg(group_names[g], metric_dict), len(group_names[g])) for g in ALL_GROUPS],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n{title}")
    print(f"  {'group':<28} {metric_label:>10}  n")
    print(f"  {'-'*28} {'-'*10}  -")
    for g, avg, n in rows:
        print(f"  {g:<28} {avg:>10.2f}  ({n})")

def print_sub_comparison(title, sub_pairs, metric_dict, metric_label):
    # sort pairs by metric descending
    rows = sorted(
        [(label, group_avg(names, metric_dict), len(names)) for label, names in sub_pairs],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n{title}")
    print(f"  {'dimension':<30} {metric_label:>10}  n")
    print(f"  {'-'*30} {'-'*10}  -")
    for label, avg, n in rows:
        print(f"  {label:<30} {avg:>10.2f}  {n}")

print("\n" + SEP)
print("📊  BENCHMARK SUMMARY")
print(SEP)

# 1. Global rankings (speed + length only)
print_ranking("🏁  SPEED RANKING — ALL (tok/s)",   speed_ranking,  "tok/s")
print_ranking("📏  LENGTH RANKING — ALL (tokens)", length_ranking, "tokens")

# 2. Group averages
print("\n" + SEP)
print("📋  GROUP AVERAGES")
print(SEP)
print_group_summary("⚡  Avg Speed by group (tok/s)",   avg_speed,  "tok/s")
print_group_summary("📐  Avg Length by group (tokens)", avg_length, "tokens")

# 3. Sub-dimension comparisons
print("\n" + SEP)
print("🔬  SUB-DIMENSION COMPARISONS")
print(SEP)

pairs_es = [("early_stop",     sub_early_stop),     ("non_early_stop", sub_non_early_stop)]
pairs_pg = [("non_paged",      sub_non_paged),       ("paged",          sub_paged)]
pairs_kv = [("no_cache gen1-4", sub_no_cache),       ("kv_cache gen5-8", sub_kv_cache)]
pairs_sf = [("score_fn1 gen1,2,5,6", sub_score1),   ("score_fn2 gen3,4,7,8", sub_score2)]

print_sub_comparison("⚡  Speed: early_stop vs non_early_stop",  pairs_es, avg_speed,  "tok/s")
print_sub_comparison("📐  Length: early_stop vs non_early_stop", pairs_es, avg_length, "tokens")

print_sub_comparison("⚡  Speed: paged vs non_paged",            pairs_pg, avg_speed,  "tok/s")
print_sub_comparison("📐  Length: paged vs non_paged",           pairs_pg, avg_length, "tokens")

print_sub_comparison("⚡  Speed: no_cache vs kv_cache",          pairs_kv, avg_speed,  "tok/s")
print_sub_comparison("📐  Length: no_cache vs kv_cache",         pairs_kv, avg_length, "tokens")

print_sub_comparison("⚡  Speed: score_fn1 vs score_fn2",        pairs_sf, avg_speed,  "tok/s")
print_sub_comparison("📐  Length: score_fn1 vs score_fn2",       pairs_sf, avg_length, "tokens")

# ================= SAVE MARKDOWN =================
MD = "/Users/thongbui.nd/Documents/Thong Bui/dev_2/src/test/benchmark_results.md"

def md_ranking_table(items, col):
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    lines = [f"| # | name | {col} |", "|---|------|------|"]
    for i, (name, val) in enumerate(sorted_items, 1):
        lines.append(f"| {i} | `{name}` | {val:.2f} |")
    return "\n".join(lines)

def md_group_summary_table(metric_dict, col):
    rows = sorted(
        [(g, group_avg(group_names[g], metric_dict), len(group_names[g])) for g in ALL_GROUPS],
        key=lambda x: x[1], reverse=True
    )
    lines = [f"| group | {col} | n |", "|-------|------|---|"]
    for g, avg, n in rows:
        lines.append(f"| `{g}` | {avg:.2f} | {n} |")
    return "\n".join(lines)

def md_sub_comparison_table(sub_pairs, metric_dict, col):
    rows = sorted(
        [(label, group_avg(names, metric_dict), len(names)) for label, names in sub_pairs],
        key=lambda x: x[1], reverse=True
    )
    lines = [f"| dimension | {col} | n |", "|-----------|------|---|"]
    for label, avg, n in rows:
        lines.append(f"| {label} | {avg:.2f} | {n} |")
    return "\n".join(lines)

with open(MD, "w", encoding="utf-8") as f:

    # ── Raw results ──────────────────────────────────────────────────────────
    f.write("# 📝 Raw Results\n\n")
    for case in all_results:
        f.write(f"## INPUT: {case['input']}\n\n")
        f.write("| name | output | time (s) | tokens | tok/s |\n")
        f.write("|------|--------|----------|--------|-------|\n")
        for r in case["runs"]:
            out_escaped = r["output"].replace("|", "\\|")
            f.write(f"| `{r['name']}` | {out_escaped} | {r['time']:.3f} | {r['tokens']} | {r['tps']:.2f} |\n")
        f.write("\n---\n\n")

    # ── Global rankings ──────────────────────────────────────────────────────
    f.write("# 🏁 Global Rankings\n\n")
    f.write("## Speed — all (tok/s)\n\n")
    f.write(md_ranking_table(speed_ranking, "tok/s"))
    f.write("\n\n## Length — all (tokens)\n\n")
    f.write(md_ranking_table(length_ranking, "tokens"))
    f.write("\n\n")

    # ── Group averages ───────────────────────────────────────────────────────
    f.write("# 📋 Group Averages\n\n")
    f.write("## Avg Speed by group (tok/s)\n\n")
    f.write(md_group_summary_table(avg_speed, "tok/s"))
    f.write("\n\n## Avg Length by group (tokens)\n\n")
    f.write(md_group_summary_table(avg_length, "tokens"))
    f.write("\n\n")

    # ── Sub-dimension comparisons ────────────────────────────────────────────
    f.write("# 🔬 Sub-Dimension Comparisons\n\n")

    for section_title, pairs in [
        ("early_stop vs non_early_stop", pairs_es),
        ("paged vs non_paged",           pairs_pg),
        ("no_cache vs kv_cache",         pairs_kv),
        ("score_fn1 vs score_fn2",       pairs_sf)
    ]:
        f.write(f"## {section_title}\n\n")
        f.write("### Speed (tok/s)\n\n")
        f.write(md_sub_comparison_table(pairs, avg_speed,  "tok/s"))
        f.write("\n\n### Length (tokens)\n\n")
        f.write(md_sub_comparison_table(pairs, avg_length, "tokens"))
        f.write("\n\n")

print(f"\n✅  Đã lưu kết quả vào: {MD}")
