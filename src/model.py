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

        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        cos_freqs = torch.repeat_interleave(cos_freqs, 2, dim=-1)
        sin_freqs = torch.repeat_interleave(sin_freqs, 2, dim=-1)

        self.register_buffer('cos_cached', cos_freqs)
        self.register_buffer('sin_cached', sin_freqs)

    def forward(self, seq_len):
        cos_freqs = self.cos_cached[:seq_len, :]
        sin_freqs = self.sin_cached[:seq_len, :]

        cos_freqs = cos_freqs.unsqueeze(0)
        sin_freqs = sin_freqs.unsqueeze(0)

        return cos_freqs, sin_freqs

    def apply_rope(self, x, cos_freqs, sin_freqs):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        cos_half = cos_freqs[..., ::2]
        sin_half = sin_freqs[..., ::2]

        rotated_x_even = x_even * cos_half - x_odd * sin_half
        rotated_x_odd = x_even * sin_half + x_odd * cos_half

        rotated_x = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        rotated_x = rotated_x.reshape(x.shape)

        return rotated_x

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

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = torch.where(mask == 0, torch.tensor(-1e9), torch.tensor(0.0))
        self.register_buffer('causal_mask', mask)

    def forward(self, x, pad_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k)

        cos_freqs, sin_freqs = self.rope(seq_len)
        cos_freqs = cos_freqs.view(1, seq_len, 1, self.d_k)
        sin_freqs = sin_freqs.view(1, seq_len, 1, self.d_k)

        q = self.rope.apply_rope(q, cos_freqs, sin_freqs)
        k = self.rope.apply_rope(k, cos_freqs, sin_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )

        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + causal_mask

        if pad_mask is not None:
            pad_mask = pad_mask.float().unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - pad_mask) * -1e9

        attention_weights = functional.softmax(scores, dim=-1)

        if self.training and self.dropout_rate > 0:
            attention_weights = functional.dropout(attention_weights, p=self.dropout_rate)

        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

        return self.wo(attention_output)

    def prefill(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        cos_freqs, sin_freqs = self.rope(seq_len)
        cos_freqs = cos_freqs.view(1, seq_len, 1, self.d_k)
        sin_freqs = sin_freqs.view(1, seq_len, 1, self.d_k)

        q = self.rope.apply_rope(q, cos_freqs, sin_freqs)
        k = self.rope.apply_rope(k, cos_freqs, sin_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        present_kv = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + causal_mask

        attn_w = functional.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, v).transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)

        return self.wo(out), present_kv

    def forward_with_cache(self, x, past_kv, cache_len):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        cos_full, sin_full = self.rope(cache_len + seq_len)
        cos_new = cos_full[:, cache_len:cache_len + seq_len, :].view(1, seq_len, 1, self.d_k)
        sin_new = sin_full[:, cache_len:cache_len + seq_len, :].view(1, seq_len, 1, self.d_k)

        q = self.rope.apply_rope(q, cos_new, sin_new)
        k = self.rope.apply_rope(k, cos_new, sin_new)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        past_kv[0][:batch_size, :, cache_len:cache_len + seq_len, :] = k
        past_kv[1][:batch_size, :, cache_len:cache_len + seq_len, :] = v

        k_full = past_kv[0][:batch_size, :, :cache_len + seq_len, :]
        v_full = past_kv[1][:batch_size, :, :cache_len + seq_len, :]

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_w = functional.softmax(scores, dim=-1)

        out = torch.matmul(attn_w, v_full).transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)

        return self.wo(out)
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        attn_output = self.mha(x, pad_mask=pad_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def prefill(self, x):
        attn_out, present_kv = self.mha.prefill(x)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(x + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        return self.layernorm2(out1 + ffn_out), present_kv

    def forward_with_cache(self, x, past_kv, cache_len):
        attn_out = self.mha.forward_with_cache(x, past_kv, cache_len)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(x + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        return self.layernorm2(out1 + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
        })
        return config

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

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.dropout_layer = nn.Dropout(dropout)
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, attention_mask=None):
        pad_mask = (inputs != 0).float() if attention_mask is None else attention_mask

        x = self.token_embedding(inputs)
        x = self.dropout_layer(x)

        for block in self.decoder_blocks:
            x = block(x, pad_mask=pad_mask)

        return self.final_layer(x)

    def forward_hidden(self, inputs, attention_mask=None):
        """
        ở đầu vào, mỗi token được biểu diễn bởi 1 vector d_model chiều, sau đó nối lại thành 1 seq, 
        quá trình đi qua transformer thì forward_hidden sẽ trả về 1 vector có seq_len phần tử, 
        mỗi phần tử là 1 vector d_model chiều mô tả mối quan hệ, ngữ nghĩa của token đó ở trong câu
        """
        if attention_mask is None:
            pad_mask = (inputs != 0).float()
        else:
            pad_mask = attention_mask
        
        x = self.token_embedding(inputs)
        x = self.dropout_layer(x)
        
        for block in self.decoder_blocks:
            x = block(x, pad_mask=pad_mask)
        return x

    def prefill(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.dropout_layer(x)

        kv_cache = []
        for block in self.decoder_blocks:
            x, present_kv = block.prefill(x)
            kv_cache.append(present_kv)

        first_logits = self.final_layer(x)[:, -1, :]
        return first_logits, kv_cache

    def decode_step(self, token_id, kv_cache):
        x = self.token_embedding(
            torch.tensor([[token_id]], dtype=torch.long, device=self.final_layer.weight.device)
        )

        new_kv_cache = []
        for block, past_kv in zip(self.decoder_blocks, kv_cache):
            x, present_kv = block.forward_with_cache(x, past_kv)
            new_kv_cache.append(present_kv)

        logits = self.final_layer(x)[:, 0, :]
        return logits, new_kv_cache

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_rate,
        })
        return config

    def generate_response(self, user_input, tokenizer, **kwargs):
        return generate(self, user_input, tokenizer, **kwargs)
