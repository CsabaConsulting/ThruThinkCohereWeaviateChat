from cohere.client import Chat, Client
from config import defaults
from langchain.llms import Cohere
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Weaviate
from operator import itemgetter


# RAG Fusion logics
# Step 1: Generate query variations
def generate_variations(query: str, variation_count: int, llm: Cohere) -> list[str]:
    # Step 1: Generate query variations:
    variation_system_prompt = """Your task is to generate {variation_count} different search queries that aim to answer the user question from multiple perspectives.
The user questions are focused on ThruThink budgeting analysis and projection web application usage, or a wide range of budgeting and accounting topics, including EBITDA, cash flow balance, inventory management, and more.
Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.
Each query MUST be in one line and one line only. You SHOULD NOT include any preamble or explanations, and you SHOULD NOT answer the questions or add anything else, just geenrate the queries."""
    variation_user_command_prompt_template = "Original question: {query}"
    variation_user_example_prompt_template = "Example output:\n"
    for i in range(variation_count):
        variation_user_example_prompt_template += f"{i + 1}. Query variation\n"

    variation_user_output_prompt_template = "OUTPUT ({variation_count} numbered queries):"
    variation_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(variation_system_prompt),
        HumanMessagePromptTemplate.from_template(variation_user_command_prompt_template),
        HumanMessagePromptTemplate.from_template(variation_user_example_prompt_template),
        HumanMessagePromptTemplate.from_template(variation_user_output_prompt_template)
    ])
    variation_chain = (
        {
            "query": itemgetter("query"),
            "variation_count": itemgetter("variation_count")
        }
        | variation_prompt
        | llm
        | StrOutputParser()
    )
    query_variations = []
    for t in range(defaults["max_retries"]):
        query_variations = variation_chain.invoke(dict(query=query, variation_count=variation_count))
        # print(f"{t}.: {query_variations}")
        if query_variations.count(".") >= variation_count and query_variations.count("\n") >= variation_count - 1:
            break

    return query_variations


def extract_query_variations(query: str, query_variations: list[str], variation_count: int) -> list[str]:
    queries = [query]
    if query_variations.count(".") >= variation_count:
        for query_variation in query_variations.split("\n")[:variation_count]:
            dot_index = query_variation.index(".") if "." in query_variation else -1
            q = query_variation[dot_index + 1:].strip()
            if q not in queries:
                queries.append(q)

    return queries


# Step 2: Retrieve documents for each query variation
def retrieve_documents_for_query_variations(queries: list[str], vectorstore: Weaviate, document_k: int) -> list[list[Document]]:
    document_sets = []
    for q in queries:
        document_sets.append(vectorstore.similarity_search_by_text(q, k=document_k))

    return document_sets


# Step 3: Rerank the document sets with reciprocal rank fusion
def rerank_and_fuse_documents(document_sets: list[list[Document]], rerank_k: int) -> list[tuple[Document, float]]:
    fused_scores = dict()
    doc_map = dict()
    for doc_set in document_sets:
        for rank, doc in enumerate(doc_set):
            title = doc.metadata["title"]
            if title not in doc_map:
                doc_map[title] = doc

            if title not in fused_scores:
                fused_scores[title] = 0

            fused_scores[title] += 1 / (rank + rerank_k)

    # reranked documents
    return [
        (doc_map[title], score)
        for title, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


# Step 4: Prepare and executing final RAG calls
# (a document based and a web connector based - also augmented)
def final_rag_operations(
    query: str,
    reranked_results: list[tuple[Document, float]],
    top_k_augment_doc: int,
    cohere_fusion_model: str,
    temperature: float,
    conversation_id: str,
    co: Client,
) -> tuple[Chat, Chat]:
    # Step 5: Cohere rerank
    documents_to_cohere_rank = []    
    for rrr in reranked_results:
        documents_to_cohere_rank.append(rrr[0].page_content)

    cohere_ranks = co.rerank(
        query=query,
        documents=documents_to_cohere_rank,
        max_chunks_per_doc=100,
        top_n=top_k_augment_doc,
        model="rerank-english-v2.0"
    )

    # Step 6: Prepare prompt augmentation for RAG
    context = ""
    documents = []
    for index, cohere_rank in enumerate(cohere_ranks):
        if context:
            context += "\n"

        rrr = reranked_results[cohere_rank.index]
        context_content = rrr[0].page_content  # .replace("\n", " ")
        context += f"{index + 1}. context: `{context_content}`"
        documents.append(dict(
            id=rrr[0].metadata["slug"],
            title=rrr[0].metadata["title"],
            category=rrr[0].metadata["category"],
            snippet=rrr[0].page_content,
        ))

    # Step 7: Final augmented RAG calls
    chat_system_prompt = """You are an assistant specialized in ThruThink budgeting analysis and projection web application usage.
You are also knowledgeable in a wide range of budgeting and accounting topics, including EBITDA, cash flow balance, inventory management, and more.
While you strive to provide accurate information and assistance, please keep in mind that you are not a licensed investment advisor, financial advisor, or tax advisor.
Therefore, you cannot provide personalized investment advice, financial planning, or tax guidance.
You are here to assist with ThruThink-related inquiries, or offer general information, answer questions to the best of your knowledge.
When provided, factor in any pieces of retrieved context to answer the question. Also factor in any
If you don't know the answer, just say that "I don't know", don't try to make up an answer."""

    rag_query = f"""Use the following pieces of retrieved context to answer the question.
---
Contexts: {context}
---
Question: {query}
Answer:
"""
    web_response = co.chat(
        model=cohere_fusion_model,
        prompt_truncation="auto",
        temperature=temperature,
        connectors=[{"id": "web-search"}],
        citation_quality="accurate",
        conversation_id=conversation_id,
        preamble_override=chat_system_prompt,
        message=rag_query,
    )

    tt_response = co.chat(
        model=cohere_fusion_model,
        prompt_truncation="auto",
        temperature=temperature,
        citation_quality="accurate",
        conversation_id=conversation_id,
        documents=documents,
        preamble_override=chat_system_prompt,
        message=query,
    )

    return tt_response, web_response
