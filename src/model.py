# import torch
# import torch.nn as nn
# import torch.nn.functional as functional
# from utils.generate import generate

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super().__init__()

#         self.d_model = d_model
#         self.max_seq_len = max_seq_len

#         inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
#         self.register_buffer('inv_freq', inv_freq)

#         position = torch.arange(max_seq_len, dtype=torch.float32)
#         freqs = torch.einsum('i,j->ij', position, inv_freq)

#         self.register_buffer('cos_cached', torch.cos(freqs))
#         self.register_buffer('sin_cached', torch.sin(freqs))

#     def apply_rope(self, x, positions):
#         cos = self.cos_cached[positions]
#         sin = self.sin_cached[positions]

#         cos = cos.unsqueeze(0).unsqueeze(0)
#         sin = sin.unsqueeze(0).unsqueeze(0)

#         x_even = x[..., ::2]
#         x_odd  = x[..., 1::2]

#         rotated_even = x_even * cos - x_odd * sin
#         rotated_odd  = x_even * sin + x_odd * cos

#         rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
#         return rotated.reshape(x.shape)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, max_seq_len, dropout_rate):
#         super().__init__()

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.max_seq_len = max_seq_len
#         self.dropout_rate = dropout_rate

#         self.wq = nn.Linear(d_model, d_model)
#         self.wk = nn.Linear(d_model, d_model)
#         self.wv = nn.Linear(d_model, d_model)
#         self.wo = nn.Linear(d_model, d_model)

#         self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

#         causal_mask = torch.triu(
#             torch.full((max_seq_len, max_seq_len), float('-inf')),
#             diagonal=1
#         )
#         self.register_buffer('causal_mask', causal_mask)

#     def forward(self, x, pad_mask=None):
#         batch_size, seq_len, _ = x.shape

#         q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

#         positions = torch.arange(seq_len, device=x.device)
#         q = self.rope.apply_rope(q, positions)
#         k = self.rope.apply_rope(k, positions)

#         attn_mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)

#         if pad_mask is not None:
#             key_pad = torch.zeros(batch_size, 1, 1, seq_len, device=x.device, dtype=x.dtype)
#             key_pad = key_pad.masked_fill(~pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
#             attn_mask = attn_mask + key_pad

#         dropout_p = self.dropout_rate if self.training else 0.0
#         out = functional.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_mask,
#             dropout_p=dropout_p,
#         )

#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
#         return self.wo(out)

#     def prefill(self, x):
#         batch_size, seq_len, _ = x.shape

#         assert seq_len <= self.max_seq_len, (
#             f"prefill seq_len ({seq_len}) vượt max_seq_len ({self.max_seq_len})"
#         )

#         q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

#         positions = torch.arange(seq_len, device=x.device)
#         q = self.rope.apply_rope(q, positions)
#         k = self.rope.apply_rope(k, positions)

#         present_kv = (k, v)

#         causal_mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
#         out = functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
#         return self.wo(out), present_kv

#     def forward_with_cache(self, x, past_kv, cache_len):
#         batch_size, seq_len, _ = x.shape
#         max_gen_len = past_kv[0].size(2)

#         assert cache_len + seq_len <= self.max_seq_len, (
#             f"RoPE overflow: cache_len({cache_len}) + seq_len({seq_len}) "
#             f"> max_seq_len({self.max_seq_len})"
#         )
#         assert cache_len + seq_len <= max_gen_len, (
#             f"KV cache overflow: cache_len({cache_len}) + seq_len({seq_len}) "
#             f"> max_gen_len({max_gen_len})"
#         )

#         q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

#         positions = torch.arange(cache_len, cache_len + seq_len, device=x.device)
#         q = self.rope.apply_rope(q, positions)
#         k = self.rope.apply_rope(k, positions)

#         past_kv[0][:batch_size, :, cache_len:cache_len + seq_len, :] = k
#         past_kv[1][:batch_size, :, cache_len:cache_len + seq_len, :] = v

#         k_full = past_kv[0][:batch_size, :, :cache_len + seq_len, :]
#         v_full = past_kv[1][:batch_size, :, :cache_len + seq_len, :]

#         seq_total = cache_len + seq_len
#         attn_mask = self.causal_mask[:seq_total, :seq_total]
#         attn_mask = attn_mask[cache_len:cache_len + seq_len, :].view(1, 1, seq_len, seq_total)

#         out = functional.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

#         return self.wo(out)
    
#     def forward_with_full_kv(self, x, k_full, v_full):
#         """
#         Attention với KV đã được gather từ PagedKVPool.
 
#         x      : (batch, 1, d_model)         — query token mới nhất
#         k_full : (batch, num_heads, seq_len, d_k)   — key của toàn bộ context
#         v_full : (batch, num_heads, seq_len, d_k)   — value của toàn bộ context
 
#         Không ghi vào cache, không copy — chỉ đọc k_full/v_full đã gather sẵn.
#         """
#         batch_size, seq_len, _ = x.shape   # seq_len == 1
#         total_kv_len = k_full.size(2)
 
#         q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
 
#         # RoPE: query token nằm ở vị trí total_kv_len - 1
#         q_pos = torch.tensor([total_kv_len - 1], device=x.device)
#         q = self.rope.apply_rope(q, q_pos)
 
#         # k_full đã có RoPE từ lúc ghi vào pool — không áp dụng lại
 
#         # Ghép k/v mới vào cuối k_full/v_full để tính attention
#         # (token mới chưa có trong pool ở thời điểm gather)
#         k_ctx = torch.cat([k_full, k], dim=2)   # (B, H, total_kv_len+1, d_k)
#         v_ctx = torch.cat([v_full, v], dim=2)
 
#         # Causal mask: query chỉ attend đến toàn bộ context (luôn visible)
#         # → không cần mask vì query là token cuối cùng
#         out = functional.scaled_dot_product_attention(q, k_ctx, v_ctx, attn_mask=None)
 
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
#         return self.wo(out)

# class SwiGLU(nn.Module):
#     """FFN với SwiGLU activation: SwiGLU(x) = (xW + b) * SiLU(xV + c)"""
#     def __init__(self, d_model, ff_dim):
#         super().__init__()
#         self.gate_proj = nn.Linear(d_model, ff_dim)
#         self.up_proj   = nn.Linear(d_model, ff_dim)
#         self.down_proj = nn.Linear(ff_dim,  d_model)

#     def forward(self, x):
#         return self.down_proj(self.up_proj(x) * functional.silu(self.gate_proj(x)))

# class DecoderBlock(nn.Module):
#     def __init__(self, d_model, num_heads, ff_dim, max_seq_len, dropout):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.max_seq_len = max_seq_len
#         self.dropout = dropout

#         self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)

#         self.ffn = SwiGLU(d_model, ff_dim)

#         self.layernorm1 = nn.RMSNorm(d_model, eps=1e-6)
#         self.layernorm2 = nn.RMSNorm(d_model, eps=1e-6)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x, pad_mask=None):
#         attn_output = self.mha(self.layernorm1(x), pad_mask=pad_mask)
#         attn_output = self.dropout1(attn_output)
#         out1 = x + attn_output

#         ffn_output = self.ffn(self.layernorm2(out1))
#         ffn_output = self.dropout2(ffn_output)
#         return out1 + ffn_output

#     def prefill(self, x):
#         attn_out, present_kv = self.mha.prefill(self.layernorm1(x))
#         attn_out = self.dropout1(attn_out)
#         out1 = x + attn_out
#         ffn_out = self.ffn(self.layernorm2(out1))
#         ffn_out = self.dropout2(ffn_out)
#         return out1 + ffn_out, present_kv

#     def forward_with_cache(self, x, past_kv, cache_len):
#         attn_out = self.mha.forward_with_cache(self.layernorm1(x), past_kv, cache_len)
#         attn_out = self.dropout1(attn_out)
#         out1 = x + attn_out
#         ffn_out = self.ffn(self.layernorm2(out1))
#         ffn_out = self.dropout2(ffn_out)
#         return out1 + ffn_out
    
#     def forward_with_full_kv(self, x, k_full, v_full):
#         attn_out = self.mha.forward_with_full_kv(self.layernorm1(x), k_full, v_full)
#         attn_out = self.dropout1(attn_out)
#         out1 = x + attn_out
#         ffn_out = self.ffn(self.layernorm2(out1))
#         ffn_out = self.dropout2(ffn_out)
#         return out1 + ffn_out


# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.ff_dim = ff_dim
#         self.max_seq_len = max_seq_len
#         self.dropout_rate = dropout

#         self.token_embedding = nn.Embedding(vocab_size, d_model)

#         assert d_model % num_heads == 0, f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads})"

#         self.decoder_blocks = nn.ModuleList([
#             DecoderBlock(d_model, num_heads, ff_dim, max_seq_len, dropout)
#             for _ in range(num_layers)
#         ])

#         self.dropout_layer = nn.Dropout(dropout)
#         self.final_norm = nn.RMSNorm(d_model, eps=1e-6)
#         self.final_layer = nn.Linear(d_model, vocab_size, bias=False)

#         nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model ** -0.5)
#         self.final_layer.weight = self.token_embedding.weight

#     def forward(self, inputs, attention_mask=None):
#         pad_mask = (inputs != 0) if attention_mask is None else attention_mask.bool()

#         x = self.token_embedding(inputs)
#         x = self.dropout_layer(x)

#         for block in self.decoder_blocks:
#             x = block(x, pad_mask=pad_mask)

#         return self.final_layer(self.final_norm(x))

#     def forward_hidden(self, inputs, attention_mask=None):
#         if attention_mask is None:
#             pad_mask = (inputs != 0)
#         else:
#             pad_mask = attention_mask.bool()
        
#         x = self.token_embedding(inputs)
#         x = self.dropout_layer(x)
        
#         for block in self.decoder_blocks:
#             x = block(x, pad_mask=pad_mask)
#         return self.final_norm(x)

#     def init_cache(self, batch_size, max_gen_len, device):
#         d_k = self.d_model // self.num_heads
#         cache = []
#         for _ in self.decoder_blocks:
#             k_buf = torch.empty(batch_size, self.num_heads, max_gen_len, d_k, device=device)
#             v_buf = torch.empty_like(k_buf)
#             cache.append((k_buf, v_buf))
#         return cache

#     def prefill(self, input_ids, kv_cache=None):
#         x = self.token_embedding(input_ids)
#         x = self.dropout_layer(x)

#         seq_len = input_ids.size(1)
#         new_cache = []
#         for i, block in enumerate(self.decoder_blocks):
#             x, present_kv = block.prefill(x)
#             if kv_cache is not None:
#                 kv_cache[i][0][:, :, :seq_len, :] = present_kv[0]
#                 kv_cache[i][1][:, :, :seq_len, :] = present_kv[1]
#                 new_cache.append(kv_cache[i])
#             else:
#                 new_cache.append(present_kv)

#         first_logits = self.final_layer(self.final_norm(x))[:, -1, :]
#         return first_logits, new_cache

#     def decode_step(self, token_ids, kv_cache, cache_len):
#         if not isinstance(token_ids, torch.Tensor):
#             token_ids = torch.tensor([token_ids], dtype=torch.long, device=self.final_layer.weight.device)
#         token_ids = token_ids.view(-1, 1)

#         x = self.token_embedding(token_ids)

#         for block, past_kv in zip(self.decoder_blocks, kv_cache):
#             x = block.forward_with_cache(x, past_kv, cache_len)

#         logits = self.final_layer(self.final_norm(x))[:, 0, :]
#         return logits, kv_cache
    
#     def forward_with_gathered_kv(self, x_tok, gathered_kvs):
#         x = self.token_embedding(x_tok)
#         x = self.dropout_layer(x)
 
#         for block, (k_full, v_full) in zip(self.decoder_blocks, gathered_kvs):
#             x = block.forward_with_full_kv(x, k_full, v_full)
 
#         return self.final_layer(self.final_norm(x))   # (1, 1, vocab_size)

#     def generate_response(self, user_input, tokenizer, **kwargs):
#         return generate(self, user_input, tokenizer, **kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generate import generate

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE đúng chuẩn: rotate_half variant.
    d_model ở đây là d_k (head dim), không phải d_model toàn cục.
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim phải chẵn"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)          # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x         : (B, num_heads, T, head_dim)
        positions : (T,) — vị trí tuyệt đối
        """
        cos = self.cos_cached[positions][None, None, :, :]  # (1,1,T,head_dim)
        sin = self.sin_cached[positions][None, None, :, :]
        return x * cos + self._rotate_half(x) * sin

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

    def _project_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def _merge(self, out: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = out.shape
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k))

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        pos = torch.arange(T, device=x.device)
        q = self.rope.apply_rope(q, pos)
        k = self.rope.apply_rope(k, pos)

        dropout_p = self.dropout_rate if self.training else 0.0

        # Tự build causal mask dạng additive float
        causal = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )  # (T, T)
        attn_mask = causal[None, None, :, :]  # (1, 1, T, T)

        if pad_mask is not None:
            pad = torch.zeros(B, 1, 1, T, device=x.device)
            pad.masked_fill_(~pad_mask[:, None, None, :], float('-inf'))
            attn_mask = attn_mask + pad  # (B, 1, T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,   # ← quan trọng: False vì đã có mask thủ công
            dropout_p=dropout_p,
        )
        return self._merge(out)

    def prefill(self, x: torch.Tensor):
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        pos = torch.arange(T, device=x.device)
        q = self.rope.apply_rope(q, pos)
        k = self.rope.apply_rope(k, pos)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self._merge(out), (k, v)

    def forward_with_cache(self, x: torch.Tensor, past_kv, cache_len: int):
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        pos = torch.arange(cache_len, cache_len + T, device=x.device)
        q = self.rope.apply_rope(q, pos)
        k = self.rope.apply_rope(k, pos)

        # Ghi K/V mới vào slot tương ứng trong pre-allocated buffer
        past_kv[0][:B, :, cache_len:cache_len + T, :] = k
        past_kv[1][:B, :, cache_len:cache_len + T, :] = v

        k_full = past_kv[0][:B, :, :cache_len + T, :]
        v_full = past_kv[1][:B, :, :cache_len + T, :]

        # Với Q=(B,H,1,d_k) và K=(B,H,S,d_k): token cuối luôn attend toàn bộ
        # → không cần causal mask thêm (query là token mới nhất)
        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        return self._merge(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ff_dim, bias=False)
        self.up   = nn.Linear(d_model, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim,  d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.mha  = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        self.ffn  = SwiGLU(d_model, ff_dim)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.mha(self.norm1(x), pad_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def prefill(self, x):
        attn, kv = self.mha.prefill(self.norm1(x))
        x = x + self.drop(attn)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, kv

    def forward_with_cache(self, x, kv, cache_len: int):
        x = x + self.drop(self.mha.forward_with_cache(self.norm1(x), kv, cache_len))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        d_model:     int,
        num_heads:   int,
        num_layers:  int,
        ff_dim:      int,
        max_seq_len: int,
        dropout:     float,
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.norm    = nn.RMSNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        pad_mask = (input_ids != 0) if attention_mask is None else attention_mask.bool()

        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, pad_mask)

        return self.lm_head(self.norm(x))

    def init_cache(self, batch_size: int, max_gen_len: int, device: torch.device):
        """Khởi tạo pre-allocated KV cache buffer."""
        d_k = self.d_model // self.num_heads
        return [
            (
                torch.empty(batch_size, self.num_heads, max_gen_len, d_k, device=device),
                torch.empty(batch_size, self.num_heads, max_gen_len, d_k, device=device),
            )
            for _ in self.blocks
        ]

    def prefill(self, input_ids: torch.Tensor, kv_cache=None):
        """
        Chạy forward pass cho toàn bộ prompt.
        Trả về logits của token cuối và KV cache đã được điền.
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)

        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block.prefill(x)
            if kv_cache is not None:
                kv_cache[i][0][:B, :, :T, :] = kv[0]
                kv_cache[i][1][:B, :, :T, :] = kv[1]
                new_cache.append(kv_cache[i])
            else:
                new_cache.append(list(kv))

        logits = self.lm_head(self.norm(x))[:, -1, :]
        return logits, new_cache

    def decode_step(self, token_ids: torch.Tensor, kv_cache, cache_len: int):
        """
        Decode một token.
        token_ids : (B,) hoặc (B, 1)
        Trả về logits (B, vocab_size).
        """
        token_ids = token_ids.view(-1, 1)
        x = self.embed(token_ids)

        for block, kv in zip(self.blocks, kv_cache):
            x = block.forward_with_cache(x, kv, cache_len)

        return self.lm_head(self.norm(x))[:, 0, :]
    
    def generate_response(self, user_input, tokenizer, **kwargs):
        return generate(self, user_input, tokenizer, **kwargs)
