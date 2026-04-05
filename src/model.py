import torch
import torch.nn as nn
import torch.nn.functional as functional
from src.utils.generate import generate

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.register_buffer('inv_freq', inv_freq)

        position = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', position, inv_freq)

        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))

    def apply_rope(self, x, positions):
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos

        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rotated.reshape(x.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout_rate):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

        causal_mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float('-inf')),
            diagonal=1
        )
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, pad_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        positions = torch.arange(seq_len, device=x.device)
        q = self.rope.apply_rope(q, positions)
        k = self.rope.apply_rope(k, positions)

        attn_mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)

        if pad_mask is not None:
            key_pad = torch.zeros(batch_size, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            key_pad = key_pad.masked_fill(~pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_mask = attn_mask + key_pad

        dropout_p = self.dropout_rate if self.training else 0.0
        out = functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.wo(out)

    def prefill(self, x):
        batch_size, seq_len, _ = x.shape

        assert seq_len <= self.max_seq_len, (
            f"prefill seq_len ({seq_len}) vượt max_seq_len ({self.max_seq_len})"
        )

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        positions = torch.arange(seq_len, device=x.device)
        q = self.rope.apply_rope(q, positions)
        k = self.rope.apply_rope(k, positions)

        present_kv = (k, v)

        causal_mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        out = functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.wo(out), present_kv

    def forward_with_cache(self, x, past_kv, cache_len):
        batch_size, seq_len, _ = x.shape
        max_gen_len = past_kv[0].size(2)

        assert cache_len + seq_len <= self.max_seq_len, (
            f"RoPE overflow: cache_len({cache_len}) + seq_len({seq_len}) "
            f"> max_seq_len({self.max_seq_len})"
        )
        assert cache_len + seq_len <= max_gen_len, (
            f"KV cache overflow: cache_len({cache_len}) + seq_len({seq_len}) "
            f"> max_gen_len({max_gen_len})"
        )

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        positions = torch.arange(cache_len, cache_len + seq_len, device=x.device)
        q = self.rope.apply_rope(q, positions)
        k = self.rope.apply_rope(k, positions)

        past_kv[0][:batch_size, :, cache_len:cache_len + seq_len, :] = k
        past_kv[1][:batch_size, :, cache_len:cache_len + seq_len, :] = v

        k_full = past_kv[0][:batch_size, :, :cache_len + seq_len, :]
        v_full = past_kv[1][:batch_size, :, :cache_len + seq_len, :]

        seq_total = cache_len + seq_len
        attn_mask = self.causal_mask[:seq_total, :seq_total]
        attn_mask = attn_mask[cache_len:cache_len + seq_len, :].view(1, 1, seq_len, seq_total)

        out = functional.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.wo(out)

class SwiGLU(nn.Module):
    """FFN với SwiGLU activation: SwiGLU(x) = (xW + b) * SiLU(xV + c)"""
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ff_dim)
        self.up_proj   = nn.Linear(d_model, ff_dim)
        self.down_proj = nn.Linear(ff_dim,  d_model)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * functional.silu(self.gate_proj(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)

        self.ffn = SwiGLU(d_model, ff_dim)

        self.layernorm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        attn_output = self.mha(self.layernorm1(x), pad_mask=pad_mask)
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        ffn_output = self.ffn(self.layernorm2(out1))
        ffn_output = self.dropout2(ffn_output)
        return out1 + ffn_output

    def prefill(self, x):
        attn_out, present_kv = self.mha.prefill(self.layernorm1(x))
        attn_out = self.dropout1(attn_out)
        out1 = x + attn_out
        ffn_out = self.ffn(self.layernorm2(out1))
        ffn_out = self.dropout2(ffn_out)
        return out1 + ffn_out, present_kv

    def forward_with_cache(self, x, past_kv, cache_len):
        attn_out = self.mha.forward_with_cache(self.layernorm1(x), past_kv, cache_len)
        attn_out = self.dropout1(attn_out)
        out1 = x + attn_out
        ffn_out = self.ffn(self.layernorm2(out1))
        ffn_out = self.dropout2(ffn_out)
        return out1 + ffn_out

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        assert d_model % num_heads == 0, f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads})"

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.dropout_layer = nn.Dropout(dropout)
        self.final_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.final_layer = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model ** -0.5)
        self.final_layer.weight = self.token_embedding.weight

        self.embed_scale = d_model ** 0.5

    def forward(self, inputs, attention_mask=None):
        pad_mask = (inputs != 0) if attention_mask is None else attention_mask.bool()

        x = self.token_embedding(inputs) * self.embed_scale
        x = self.dropout_layer(x)

        for block in self.decoder_blocks:
            x = block(x, pad_mask=pad_mask)

        return self.final_layer(self.final_norm(x))

    def forward_hidden(self, inputs, attention_mask=None):
        if attention_mask is None:
            pad_mask = (inputs != 0)
        else:
            pad_mask = attention_mask.bool()
        
        x = self.token_embedding(inputs) * self.embed_scale
        x = self.dropout_layer(x)
        
        for block in self.decoder_blocks:
            x = block(x, pad_mask=pad_mask)
        return self.final_norm(x)

    def init_cache(self, batch_size, max_gen_len, device):
        d_k = self.d_model // self.num_heads
        cache = []
        for _ in self.decoder_blocks:
            k_buf = torch.empty(batch_size, self.num_heads, max_gen_len, d_k, device=device)
            v_buf = torch.empty_like(k_buf)
            cache.append((k_buf, v_buf))
        return cache

    def prefill(self, input_ids, kv_cache=None):
        x = self.token_embedding(input_ids) * self.embed_scale
        x = self.dropout_layer(x)

        seq_len = input_ids.size(1)
        new_cache = []
        for i, block in enumerate(self.decoder_blocks):
            x, present_kv = block.prefill(x)
            if kv_cache is not None:
                kv_cache[i][0][:, :, :seq_len, :] = present_kv[0]
                kv_cache[i][1][:, :, :seq_len, :] = present_kv[1]
                new_cache.append(kv_cache[i])
            else:
                new_cache.append(present_kv)

        first_logits = self.final_layer(self.final_norm(x))[:, -1, :]
        return first_logits, new_cache

    def decode_step(self, token_ids, kv_cache, cache_len):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor([token_ids], dtype=torch.long, device=self.final_layer.weight.device)
        token_ids = token_ids.view(-1, 1)

        x = self.token_embedding(token_ids) * self.embed_scale

        for block, past_kv in zip(self.decoder_blocks, kv_cache):
            x = block.forward_with_cache(x, past_kv, cache_len)

        logits = self.final_layer(self.final_norm(x))[:, 0, :]
        return logits, kv_cache

    def generate_response(self, user_input, tokenizer, **kwargs):
        return generate(self, user_input, tokenizer, **kwargs)
