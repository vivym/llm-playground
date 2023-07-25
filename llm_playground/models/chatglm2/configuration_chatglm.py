from typing import Optional

from transformers import PretrainedConfig


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers: int = 28,
        padded_vocab_size: int = 65024,
        hidden_size: int = 4096,
        ffn_hidden_size: int = 13696,
        kv_channels: int = 128,
        num_attention_heads: int = 32,
        seq_length: int = 2048,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layernorm_epsilon: float = 1e-5,
        rmsnorm: bool = True,
        apply_residual_connection_post_layernorm: bool = False,
        post_layer_norm: bool = True,
        add_bias_linear: bool = False,
        add_qkv_bias: bool = False,
        bias_dropout_fusion: bool = True,
        multi_query_attention: bool = False,
        multi_query_group_num: int = 1,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = True,
        fp32_residual_connection: bool = False,
        quantization_bit: int = 0,
        pre_seq_len: Optional[int] = None,
        prefix_projection: bool = False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        super().__init__(**kwargs)
