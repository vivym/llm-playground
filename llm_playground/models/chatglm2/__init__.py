from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from .configuration_chatglm import ChatGLMConfig
from .modeling_chatglm import ChatGLMForConditionalGeneration
from .tokenization_chatglm import ChatGLMTokenizer


def register():
    AutoConfig.register("chatglm", ChatGLMConfig)
    AutoModel.register(ChatGLMConfig, ChatGLMForConditionalGeneration)
    AutoModelForCausalLM.register(ChatGLMConfig, ChatGLMForConditionalGeneration)
    AutoModelForSeq2SeqLM.register(ChatGLMConfig, ChatGLMForConditionalGeneration)
    AutoTokenizer.register(ChatGLMConfig, slow_tokenizer_class=ChatGLMTokenizer)
