import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from llm_playground.models.chatglm2 import register

model: PreTrainedModel = None
tokenizer: PreTrainedTokenizer = None


def get_model():
    global model, tokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "./weights/chatglm2-6b-int4",
        )

    if model is None:
        model = AutoModel.from_pretrained(
            "./weights/chatglm2-6b-int4",
        )
        model = model.cuda().eval()

    return model, tokenizer


def summarize(text: str) -> str:
    model, tokenizer = get_model()

    response, _ = model.chat(tokenizer, text + "\nSummarize the article above in one sentence.", history=[])
    return response


def main():
    register()

    text1 = """We would like to congratulate the authors, comprising current and former members of the Database Systems and Information Management (DIMA) Group at TU Berlin, for the acceptance of their research paper "Apache Wayang: A Unified Data Analytics Framework" for publication in the SIGMOD Record 2023. 
    Title:
    Apache Wayang: A Unified Data Analytics Framework
    Authors:
    Kaustubh Beedkar, Bertty Contreras-Rojas, Haralampos Gavriilidis, Zoi Kaoudi, Volker Markl, Rodrigo Pardo-Meza, Jorge-Arnulfo Quian茅-Ruiz
    Abstract:
    The large variety of specialized data processing platforms and the increased complexity of data analytics has led to the need for unifying data analytics within a single framework. Such a framework should free users from the burden of (i) choosing the right platform(s) and (ii) gluing code between the different parts of their pipelines. Apache Wayang (Incubating) is the only open-source framework that provides a systematic solution to unified data analytics by integrating multiple heterogeneous data processing platforms. It achieves that by decoupling applications from the underlying platforms and providing an optimizer so that users do not have to specify the platforms on which their pipeline should run. Wayang provides a unified view and processing model, effectively integrating the hodgepodge of heterogeneous platforms into a single framework with increased usability without sacrificing performance and total cost of ownership. In this paper, we present the architecture of Wayang, describe its main components, and give an outlook on future directions."""

    text2 = """In Memoriam
Jorge-Arnulfo Quian茅-Ruiz
It is with deep sadness and regret that we must acknowledge the sudden death of our former colleague and friend Jorge-Arnulfo Quian茅-Ruiz. He is and will forever be sorely missed. Our thoughts and prayers are with his wife Zoi Kaoudi, his four children, and his loved ones.
From November 2019 to December 2022, Jorge was Project Leader of the Research Project Agora in the Chair of Database Systems and Information Management (DIMA) led by Prof. Dr. Volker Markl at the Technische Universit盲t Berlin. In 2020, he was appointed a Junior Research Group Leader in the Berlin Institute for the Foundations of Learning and Data (BIFOLD). It is with great appreciation that we honor and remember Jorge as an outstanding team member and an excellent scientist, who always displayed great enthusiasm, optimism, and drive. He was consistently cheerful, warm, open-minded, and was always willing to help others.
From 2004 through 2008, Jorge conducted research in distributed query processing as a doctoral student in Inria (France) and earned his PhD in Computer Science in 2008 from Nantes Universit茅. Over the following fifteen years, he joined Inria as a Research Engineer, the Universit盲t des Saarlandes as a Postdoctoral Researcher, Qatar Computing Research Institute (QCRI) as a Senior Researcher, the DIMA Group at TU Berlin as a Principal Researcher, the German Research Center for Artificial Intelligence (DFKI) as a Scientific Advisor, and BIFOLD as Head of the Big Data Systems Group. In January 2023, both he and his wife Zoi Kaoudi were appointed Associate Professors in the Department of Computer Science (Data-intensive Systems and Applications -- DASYA Research Group) at the IT University of Copenhagen (ITU) in Denmark.
Over the years, Jorge has received multiple awards for research on cross-platform computing and scalable infrastructures. Among them the 2022 ACM SIGMOD Research Highlight Award, the ICDE 2022 Best Demo Award, and the ICDE 2021 Best Paper Award, as well as several patents in core database areas and machine learning. Furthermore, he led and managed the Rheem Project on Big Data Management, now incubating in the Apache Software Foundation as Apache Wayang, and the Agora Project, which aims to create a data ecosystem for cross-platform computing. Moreover, he was a co-founder and CTO of databloom.ai, a startup company."""

    text3 = """Enterprises want to infuse Large Language Models (LLMs) into their mission-critical applications. However, the unpredictable nature of LLMs can lead to hallucinations 鈥 inaccurate inferences or outright errors 鈥 posing serious challenges for enterprises looking for accuracy, explainability, and reliability.
Retrieval augmented generation is the leading consideration for overcoming these challenges, by grounding your LLM in facts. Knowledge graphs and vector databases are the two primary contenders as potential solutions for implementing retrieval augmented generation. But which one of them offers a more accurate, reliable, and explainable foundation for your LLM?
Let鈥檚 take a look at some of the key factors to consider when choosing between knowledge graphs and vector databases to ground your LLM.
Answering Complex Questions

The higher the complexity of the question, the harder it is for a vector database to quickly and efficiently return results. Adding more subjects to a query makes it harder for the database to find the information you want.
For example: Both a knowledge graph and a vector database can easily return an answer to 鈥淲ho is the CEO of my company?鈥 but a knowledge graph will outpace a vector database on a question like 鈥淲hich board meetings in the last twelve months had at least two members abstain from a vote?鈥
A vector database is likely to find an answer in the middle of the subjects within the vector space, and not the specific answer. A knowledge graph looks for and returns precise information based on traversing a graph that is connected by relationships.
Getting Complete Responses

Vector databases are more likely to provide incomplete or irrelevant results when returning an answer because they rely on similarity scoring and a predefined result limit.
For example: If you ask: 鈥淟ist all the books written by John Smith,鈥 a vector database will return:
An incomplete list of titles (predefined limit too low), or
All titles by John Smith and some by other authors (predefined limit too high), or
The exact answer (predefined limit just right).
Because developers can鈥檛 know the predefined limit for all possible queries, it is nearly impossible to get an exact answer from a vector database.
However, because knowledge graph entities are directly connected by relationships, the number of relationships is different for every entity. Knowledge graphs retrieve and return the exact answer, and nothing more. In this case, a knowledge graph query will return all books written by John Smith and nothing else.
Getting Credible Responses

Vector databases can connect two factual pieces of information together and infer something inaccurate.
For example: If you asked: 鈥淲ho is on the product management team?鈥, a vector database might incorrectly infer that someone was on the product team because they have frequent commenting access to documents (fact) produced by the product team (fact) and return their name in the results. Because a knowledge graph uses nodes and relationships to identify how people in an organization are related, it would return only those on the product team.
Knowledge graph queries follow a flow of connected information, making responses consistently accurate and explainable.
Correcting LLM Hallucinations

Knowledge graphs have a human-readable representation of data, whereas vector databases offer only a black box.
For example: When a member of the product team is misidentified, a vector database will not be able to identify the facts it used to infer the misinformation. This means it isn鈥檛 possible to undo it or even understand the source of the error. On the other hand, it鈥檚 easy for knowledge graph users to find and correct the misinformation, should the LLM infer something incorrectly.
That鈥檚 because knowledge graphs have full transparency. They help you identify misinformation in data, trace back the pathway of the query, and make corrections to it, which can help improve LLM accuracy. Vector databases, on the other hand, provide little to no transparency and no ability to make specific corrections.
Knowledge Graphs for Your LLM

Knowledge graphs are the best choice to back your LLM to help ensure accuracy, explainability, and context. Neo4j鈥檚 reliable and verifiable knowledge graph boosts LLM accuracy and explainability, offering robust enterprise capabilities like data protection, governance, high availability, scalability, and flexible deployment, which make it a reliable and scalable choice to pair with LLMs that support mission-critical applications.
Learn more about Neo4j鈥檚 knowledge graphs for LLM-powered applications or read up on building knowledge graphs in the newly published O鈥橰eilly book.

Learn More About Knowledge Graphs and LLMs"""

    for text in [text1, text2, text3]:
        summary = summarize(text)
        print(summary)


if __name__ == "__main__":
    main()
