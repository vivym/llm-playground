import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from llm_playground.models.chatglm2 import register

text = """We would like to congratulate the authors, comprising current and former members of the Database Systems and Information Management (DIMA) Group at TU Berlin, for the acceptance of their research paper "Apache Wayang: A Unified Data Analytics Framework" for publication in the SIGMOD Record 2023. 
Title:
Apache Wayang: A Unified Data Analytics Framework
Authors:
Kaustubh Beedkar, Bertty Contreras-Rojas, Haralampos Gavriilidis, Zoi Kaoudi, Volker Markl, Rodrigo Pardo-Meza, Jorge-Arnulfo Quian茅-Ruiz
Abstract:
The large variety of specialized data processing platforms and the increased complexity of data analytics has led to the need for unifying data analytics within a single framework. Such a framework should free users from the burden of (i) choosing the right platform(s) and (ii) gluing code between the different parts of their pipelines. Apache Wayang (Incubating) is the only open-source framework that provides a systematic solution to unified data analytics by integrating multiple heterogeneous data processing platforms. It achieves that by decoupling applications from the underlying platforms and providing an optimizer so that users do not have to specify the platforms on which their pipeline should run. Wayang provides a unified view and processing model, effectively integrating the hodgepodge of heterogeneous platforms into a single framework with increased usability without sacrificing performance and total cost of ownership. In this paper, we present the architecture of Wayang, describe its main components, and give an outlook on future directions.
"""


def main():
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

    response, history = model.chat(tokenizer, text + "\nSummarize the article above in one sentence.", history=[])
    print(response)


if __name__ == "__main__":
    main()
