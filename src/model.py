import tensorflow as tf
from keras import layers, models

class RotaryPositionalEmbedding(layers.Layer):
    def __init__(self, d_model, max_seq_len, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (tf.range(0, d_model, 2, dtype=tf.float32) / d_model))
        self.inv_freq = tf.Variable(inv_freq, trainable=False, name="inv_freq")
        
        position = tf.range(max_seq_len, dtype=tf.float32)
        freqs = tf.einsum('i,j->ij', position, inv_freq)
        
        cos_freqs = tf.cos(freqs)
        sin_freqs = tf.sin(freqs)
        
        cos_freqs = tf.repeat(cos_freqs, 2, axis=-1)
        sin_freqs = tf.repeat(sin_freqs, 2, axis=-1)
        
        self.cos_cached = tf.Variable(cos_freqs, trainable=False, name="cos_freqs")
        self.sin_cached = tf.Variable(sin_freqs, trainable=False, name="sin_freqs")
    
    def call(self, seq_len):
        tf.debugging.assert_less_equal(
            seq_len, self.max_seq_len,
            message=f"seq_len exceeds max_seq_len ({self.max_seq_len})"
        )
        
        cos_freqs = self.cos_cached[:seq_len, :]
        sin_freqs = self.sin_cached[:seq_len, :]
        
        cos_freqs = tf.expand_dims(cos_freqs, 0)
        sin_freqs = tf.expand_dims(sin_freqs, 0)
        
        return cos_freqs, sin_freqs
    
    def apply_rope(self, x, cos_freqs, sin_freqs):
        """Apply RoPE to input tensor x"""
        x_even = x[..., ::2]  # [batch, seq_len, heads, d_k//2]
        x_odd = x[..., 1::2]  # [batch, seq_len, heads, d_k//2]
        
        cos_half = cos_freqs[..., ::2]  # [1, seq_len, 1, d_k//2]
        sin_half = sin_freqs[..., ::2]  # [1, seq_len, 1, d_k//2]
        
        rotated_x_even = x_even * cos_half - x_odd * sin_half
        rotated_x_odd = x_even * sin_half + x_odd * cos_half
        
        rotated_x = tf.stack([rotated_x_even, rotated_x_odd], axis=-1)
        rotated_x = tf.reshape(rotated_x, tf.shape(x))
        
        return rotated_x

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, max_seq_len, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        
        # Khởi tạo các lớp trọng số
        self.wq = layers.Dense(d_model, name="query")
        self.wk = layers.Dense(d_model, name="key")
        self.wv = layers.Dense(d_model, name="value")
        self.wo = layers.Dense(d_model, name="output")
        
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
         
        #
        mask = tf.linalg.band_part(tf.ones((max_seq_len, max_seq_len)), -1, 0)
        mask = tf.where(mask == 0, -1e9, 0.0)
        self.causal_mask = tf.Variable(mask, trainable=False, name="causal_mask")
    
    def call(self, x, training=False):
        # lấy batch_size và seq_len từ x
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        q = self.wq(x)  # [batch, seq_len, d_model]
        k = self.wk(x)
        v = self.wv(x)
        
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.d_k))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.d_k))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.d_k))
        
        cos_freqs, sin_freqs = self.rope(seq_len)
        cos_freqs = tf.reshape(cos_freqs, (1, seq_len, 1, self.d_k))
        sin_freqs = tf.reshape(sin_freqs, (1, seq_len, 1, self.d_k))
        
        q = self.rope.apply_rope(q, cos_freqs, sin_freqs)
        k = self.rope.apply_rope(k, cos_freqs, sin_freqs)
        
        q = tf.transpose(q, [0, 2, 1, 3])  # [batch, num_heads, seq_len, d_k]
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, tf.float32))
        
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores += causal_mask
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        if training and self.dropout_rate > 0:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout_rate)
        
        attention_output = tf.matmul(attention_weights, v)
        
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))
        
        return self.wo(attention_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout_rate,
        })
        return config

class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, max_seq_len, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.mha(x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout
        })
        return config

class Model(models.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)

        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, ff_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ]
        self.dropout_layer = layers.Dropout(dropout)
        self.final_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.token_embedding(inputs)
        x = self.dropout_layer(x, training=training)

        for block in self.decoder_blocks:
            x = block(x, training=training)

        return self.final_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
        })
        return config
