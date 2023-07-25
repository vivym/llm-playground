import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from llm_playground.models.chatglm2 import register


def main():
    # model: PreTrainedModel = AutoModel.from_pretrained(
    #     "./weights/chatglm2-6b-original",
    #     torch_dtype=torch.float16,
    #     trust_remote_code=True,
    # )
    # model.save_pretrained(
    #     "./weights/chatglm2-6b",
    #     max_shard_size="1GB",
    #     safe_serialization=True,
    # )

    register()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "./weights/chatglm2-6b"
    )

    model: PreTrainedModel = AutoModel.from_pretrained(
        "./weights/chatglm2-6b",
        torch_dtype=torch.float16,
    )
    model = model.half().cuda().eval()
    print(model)

    response, history = model.chat(tokenizer, "下面这句话是疑问句还是陈述句：“今天天气真不错！”", history=[])
    print(response)


if __name__ == "__main__":
    main()
