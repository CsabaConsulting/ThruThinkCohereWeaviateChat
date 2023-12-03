import threading

from cohere.client import Chat, Client, Reranking
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
from streamlit.runtime.scriptrunner import add_script_run_ctx, ScriptRunContext
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


# RAG Fusion logics
# Step 0: Resolve any context references if applicable
def resolve_conversational_references(query: str, questions: list[str], answers: list[str], llm: Cohere) -> str:
    # No history yet
    if not questions:
        print("first")
        return query

    reference_resolution_example_prompt = """Example conversation history:
Query: What is ThruThink?
Response: ThruThink is a software tool for businesses to project their performance and evaluate their debt and equity structure.
---
Example follow up question: And who operates or owns it?
---
The resolved example follow up query: And who operates or owns ThruThink?
"""

    reference_resolution_human_prompt = "The actual conversation history:\n"
    for question, answer in zip(questions, answers):
        reference_resolution_human_prompt += f"Query: {question}\n"
        reference_resolution_human_prompt += f"Response: {answer}\n"

    reference_resolution_human_prompt += "---\n"
    reference_resolution_human_prompt += "The actual follow up query: {query}\n"
    reference_resolution_human_prompt += "---\n"
    reference_resolution_human_prompt += "The resolved actual follow up query:"
    reference_resolution_prompt_array = [
        SystemMessagePromptTemplate.from_template("""Your task is to take a conversation plus a follow up query, and resolve any conversational references in the follow up query.
Every conversational reference should be substituted so if someone is presented with the resolved query it can stand alone without the chat history.
You MUST leave the non conversational references and any non reference parts of the query intact.
You SHOULD NOT include any preamble or explanations, and you SHOULD NOT answer any questions or add anything else, just resolve the conversational references.
The substitution MUST be as terse and compact as possible.
The resolved query MUST be plain text and CANNOT contain any HTML or other markup."""),
        HumanMessagePromptTemplate.from_template(reference_resolution_example_prompt),
        HumanMessagePromptTemplate.from_template(reference_resolution_human_prompt),
    ]
    reference_resolution_prompt = ChatPromptTemplate.from_messages(reference_resolution_prompt_array)
    print(reference_resolution_prompt)
    reference_resolution_chain = (
        dict(query=itemgetter("query"))
        | reference_resolution_prompt
        | llm
        | StrOutputParser()
    )
    resolved_query = reference_resolution_chain.invoke(dict(query=query))
    print(resolved_query)
    return resolved_query


# Step 1: Generate query variations
def generate_variations(query: str, variation_count: int, llm: Cohere, example_questions: bool) -> list[str]:
    # Step 1: Generate query variations:
    variation_prompt_array = [
        SystemMessagePromptTemplate.from_template("""Your task is to generate {variation_count} different search queries that aim to answer the user question from multiple perspectives.
The user questions are focused on ThruThink budgeting analysis and projection web application usage, or a wide range of budgeting and accounting topics, including EBITDA, cash flow balance, inventory management, and more.
Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.
Each query MUST be in one line and one line only. You SHOULD NOT include any preamble or explanations, and you SHOULD NOT answer the questions or add anything else, just generate the queries."""),
        HumanMessagePromptTemplate.from_template("Original question: {query}"),
    ]
    variation_user_example_prompt_template = "Example output:\n"
    if example_questions:
        for i in range(variation_count):
            variation_user_example_prompt_template += f"{i + 1}. Query variation\n"

        variation_prompt_array.append(HumanMessagePromptTemplate.from_template(variation_user_example_prompt_template))

    variation_prompt_array.append(HumanMessagePromptTemplate.from_template("OUTPUT ({variation_count} numbered queries):"))
    variation_prompt = ChatPromptTemplate.from_messages(variation_prompt_array)
    variation_chain = (
        dict(
            query=itemgetter("query"),
            variation_count=itemgetter("variation_count")
        )
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


def retrieve_documents_for_query_variation_func(ctx: ScriptRunContext, query: str, document_sets: list, vectorstore: Weaviate, document_k: int):
    add_script_run_ctx(ctx) # register context on thread func
    docs = vectorstore.similarity_search_by_text(query, k=document_k)
    document_sets.append(docs)


# Step 2: Retrieve documents for each query variation
def retrieve_documents_for_query_variations(queries: list[str], vectorstore: Weaviate, document_k: int) -> list[list[Document]]:
    ctx = get_script_run_ctx() # create a context
    thread_list = []
    document_sets = []
    for q in queries:
        # pass context to thread
        t = threading.Thread(target=retrieve_documents_for_query_variation_func, args=(ctx, q, document_sets, vectorstore, document_k))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()

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


# Step 4: Cohere Rerank
def cohere_reranking(
    query: str,
    reranked_results: list[tuple[Document, float]],
    top_k_augment_doc: int,
    co: Client,
) -> Reranking:
    documents_to_cohere_rank = []
    for rrr in reranked_results:
        documents_to_cohere_rank.append(rrr[0].page_content)

    return co.rerank(
        query=query,
        documents=documents_to_cohere_rank,
        max_chunks_per_doc=100,
        top_n=top_k_augment_doc,
        model="rerank-english-v2.0"
    )


def document_based_query_func(
    ctx: ScriptRunContext,
    cohere_fusion_model: str,
    temperature: float,
    conversation_id: str,
    chat_system_prompt: str,
    documents: list[dict],
    query: str,
    results: list[Chat],
    co: Client,
):
    add_script_run_ctx(ctx) # register context on thread func
    results[0] = co.chat(
        model=cohere_fusion_model,
        prompt_truncation="auto",
        temperature=temperature,
        citation_quality="accurate",
        conversation_id=conversation_id,
        documents=documents,
        preamble_override=chat_system_prompt,
        message=query,
    )


def web_connector_query_func(
    ctx: ScriptRunContext,
    cohere_fusion_model: str,
    temperature: float,
    conversation_id: str,
    chat_system_prompt: str,
    rag_query: str,
    results: list[Chat],
    co: Client,
):
    add_script_run_ctx(ctx) # register context on thread func
    results[1] = co.chat(
        model=cohere_fusion_model,
        prompt_truncation="auto",
        temperature=temperature,
        connectors=[dict(id="web-search")],
        citation_quality="accurate",
        conversation_id=conversation_id,
        preamble_override=chat_system_prompt,
        message=rag_query,
    )


# Step 5: Prepare and executing final RAG calls
# (a document based and a web connector based - also augmented)
def final_rag_operations(
    query: str,
    reranked_results: list[tuple[Document, float]],
    reranking: Reranking,
    cohere_fusion_model: str,
    temperature: float,
    conversation_id: str,
    co: Client,
) -> tuple[Chat, Chat]:
    # Step 6: Prepare prompt augmentation for RAG
    context = ""
    documents = []
    for index, cohere_rank in enumerate(reranking):
        if context:
            context += "\n"

        rrr = reranked_results[cohere_rank.index]
        context_content = rrr[0].page_content
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
    ctx = get_script_run_ctx() # create a context
    results = [None, None]
    document_based_query_thread = threading.Thread(
        target=document_based_query_func,
        args=(
            ctx,
            cohere_fusion_model,
            temperature,
            conversation_id,
            chat_system_prompt,
            documents,
            query,
            results,
            co
        )
    )
    web_connector_query_thread = threading.Thread(
        target=web_connector_query_func,
        args=(
            ctx,
            cohere_fusion_model,
            temperature,
            conversation_id,
            chat_system_prompt,
            rag_query,
            results,
            co
        )
    )

    document_based_query_thread.start()
    web_connector_query_thread.start()

    document_based_query_thread.join()
    web_connector_query_thread.join()

    return results
