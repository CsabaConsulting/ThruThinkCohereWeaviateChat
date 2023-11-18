import cohere
import os
import streamlit as st
import uuid
import weaviate

from config import defaults, boundaries
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Weaviate
from operator import itemgetter
from streamlit_chat import message

title = "ThruThink Support"
st.set_page_config(
    page_title=title,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.image("https://raw.githubusercontent.com/TTGithub/TTMarketingSite/master/img/logo2_smaller.png", width=300)

# Set API keys
weaviate_api_key = os.getenv("weaviate_api_key")
if not weaviate_api_key:
    weaviate_api_key = st.secrets["weaviate_api_key"]
    os.environ["weaviate_api_key"] = weaviate_api_key

weaviate_url = os.getenv("weaviate_url")
if not weaviate_url:
    weaviate_url = st.secrets["weaviate_url"]
    os.environ["weaviate_url"] = weaviate_url

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    cohere_api_key = st.secrets["cohere_api_key"]
    os.environ["COHERE_API_KEY"] = cohere_api_key

# Initialise session state variables
if "variation_count" not in st.session_state:
    st.session_state.variation_count = defaults["variation_count"]

if "index_name" not in st.session_state:
    st.session_state.index_name = defaults["index_name"]

if "text_key" not in st.session_state:
    st.session_state.text_key = defaults["text_key"]

if "document_k" not in st.session_state:
    st.session_state.document_k = defaults["document_k"]

if "rerank_k" not in st.session_state:
    st.session_state.rerank_k = defaults["rerank_k"]

if "top_k_augment_doc" not in st.session_state:
    st.session_state.top_k_augment_doc = defaults["top_k_augment_doc"]

if "temperature" not in st.session_state:
    st.session_state.temperature = defaults["temperature"]

if "cohere_variation_model" not in st.session_state:
    st.session_state.cohere_variation_model = defaults["cohere_variation_model"]

if "cohere_fusion_model" not in st.session_state:
    st.session_state.cohere_fusion_model = defaults["cohere_fusion_model"]

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "generated_tt" not in st.session_state:
    st.session_state["generated_tt"] = []

if "tt_citations" not in st.session_state:
    st.session_state["tt_citations"] = []

if "tt_documents" not in st.session_state:
    st.session_state["tt_documents"] = []

if "generated_web" not in st.session_state:
    st.session_state["generated_web"] = []

if "web_citations" not in st.session_state:
    st.session_state["web_citations"] = []

if "web_documents" not in st.session_state:
    st.session_state["web_documents"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Sidebar
variation_count = st.sidebar.slider(
    min_value=boundaries["variation_count_min"],
    max_value=boundaries["variation_count_max"],
    value=st.session_state.variation_count,
    step=1,
    label="Variation Count"
)
index_name = st.session_state.index_name
text_key = st.session_state.text_key
document_k = st.sidebar.slider(
    min_value=boundaries["document_k_min"],
    max_value=boundaries["document_k_max"],
    value=st.session_state.document_k,
    step=1,
    label="Top K doc (per variation)"
)
rerank_k = st.session_state.rerank_k
top_k_augment_doc = st.sidebar.slider(
    min_value=boundaries["top_k_augment_doc_min"],
    max_value=boundaries["top_k_augment_doc_max"],
    value=st.session_state.top_k_augment_doc,
    step=1,
    label="Top K doc (final fusion)"
)
temperature = st.sidebar.slider(
    min_value=boundaries["temperature_min"],
    max_value=boundaries["temperature_max"],
    value=st.session_state.temperature,
    step=0.05,
    label="LLM Temperature"
)
cohere_variation_model = st.session_state.cohere_variation_model
cohere_fusion_model = st.session_state.cohere_fusion_model
conversation_id = st.session_state.conversation_id

llm = Cohere(model=cohere_variation_model)
auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
client = weaviate.Client(
  url=weaviate_url,
  auth_client_secret=auth_config,
  additional_headers={"X-Cohere-Api-Key": cohere_api_key}
)
vectorstore = Weaviate(client, index_name, text_key)
vectorstore._query_attrs = ["text", "title", "category", "slug", "_additional {distance}"]
vectorstore.embedding = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=cohere_api_key)
co = cohere.Client(cohere_api_key)

# RAG Fusion logics
def generate_response_with_rag_fusion(query):
    # Step 1: Generate query variations:
    with st.spinner("Generating variations..."):
        variation_system_prompt = """You are a helpful assistant that generates multiple search queries based on a single input query.
    Do not include any explanations, do not repeat the queries, and do not answer the queries, simply just generate the alternative query variations."""
        variation_user_command_prompt_template = "The single input query: {query}"
        # variation_user_example_prompt_template = "Example output:"
        # for i in range(variation_count):
        #     variation_user_example_prompt_template += f"{i}. Query variation {i}?\n"

        variation_user_output_prompt_template = "Query variations:"
        variation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(variation_system_prompt),
            HumanMessagePromptTemplate.from_template(variation_user_command_prompt_template),
            # HumanMessagePromptTemplate.from_template(variation_user_example_prompt_template),
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
        for t in range(defaults["max_retries"]):
            query_variations = variation_chain.invoke(dict(query=query, variation_count=variation_count))
            # print(f"{t}.: {query_variations}")
            if query_variations.count(".") >= variation_count and query_variations.count("\n") >= variation_count - 1:
                break

        queries = [query]
        if query_variations.count(".") >= variation_count:
            for query_variation in query_variations.split("\n")[:variation_count]:
                dot_index = query_variation.index(".") if "." in query_variation else -1
                q = query_variation[dot_index + 1:].strip()
                if q not in queries:
                    queries.append(q)

    # Step 2: Retrieve documents for each query variation
    with st.spinner("Retrieving for variations and reranking..."):
        doc_sets = []
        for q in queries:
            doc_sets.append(vectorstore.similarity_search_by_text(q, k=document_k))

        doc_hash = dict()
        for doc_set in doc_sets:
            for rank, doc in enumerate(doc_set):
                title = doc.metadata["title"]
                doc_hash[title] = True

    # Step 3: Rerank the document sets with reciprocal rank fusion
    fused_scores = dict()
    doc_map = dict()
    for doc_set in doc_sets:
        for rank, doc in enumerate(doc_set):
            title = doc.metadata["title"]
            if not title in doc_map:
                doc_map[title] = doc

            if title not in fused_scores:
                fused_scores[title] = 0

            fused_scores[title] += 1 / (rank + rerank_k)

    reranked_results = [
        (doc_map[title], score)
        for title, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Step 4: Prepare and executing final RAG calls (a grounded and a web)
    with st.spinner("Executing twin queries..."):
        context = ""
        documents = []
        for index, rrr in enumerate(reranked_results[:top_k_augment_doc]):
            if context:
                context += "\n"

            context_content = rrr[0].page_content  # .replace("\n", " ")
            context += f"{index + 1}. context: `{context_content}`"
            documents.append(dict(
                id=rrr[0].metadata["slug"],
                title=rrr[0].metadata["title"],
                category=rrr[0].metadata["category"],
                snippet=rrr[0].page_content,
            ))

        # Step 5: Final RAG calls
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
            documents=documents,
            preamble_override=chat_system_prompt,
            message=query,
        )

    return tt_response, web_response


def insert_substring(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]


def mark_citations(kind, index, response):
    doc_map = dict()
    for idx, doc in enumerate(response.documents or []):
        if doc["id"] not in doc_map:
            doc_map[doc["id"]] = "#" + f"{kind}_ref_{index + 1}_{idx + 1}"

    txt = response.text or "N/A"
    for cit in (response.citations or [])[::-1]:
        doc_anchor = doc_map[cit["document_ids"][0]]
        cit_idxs = doc_anchor.split("_")
        cit_sup = f"{cit_idxs[-2]}.{cit_idxs[-1]}"
        txt = insert_substring(txt, f"<sup>{cit_sup}</sup></a>", cit["end"])
        txt = insert_substring(txt, f"<a href='{doc_anchor}' target='_self'>", cit["start"])

    return txt  # .replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")


clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button:
    del st.session_state.generated_tt[:]
    del st.session_state.tt_citations[:]
    del st.session_state.tt_documents[:]
    del st.session_state.generated_web[:]
    del st.session_state.web_citations[:]
    del st.session_state.web_documents[:]
    del st.session_state.past[:]
    conversation_id = str(uuid.uuid4())
    st.session_state.conversation_id = conversation_id


tt_tab, web_tab = st.tabs(["Documentation Guided Results", "Web Search Guided Results"])

user_input = st.chat_input(key="user_input", placeholder="Question", max_chars=None, disabled=False)

if user_input:
    user_input = st.session_state.user_input
    tt_response, web_response = generate_response_with_rag_fusion(user_input)
    conversation_index = len(st.session_state.past)
    st.session_state.past.append(user_input)
    generated = mark_citations("tt", conversation_index, tt_response)
    st.session_state.generated_tt.append(generated)
    st.session_state.tt_citations.append(tt_response.citations or [])
    st.session_state.tt_documents.append(tt_response.documents or [])
    generated = mark_citations("web", conversation_index, web_response)
    st.session_state.generated_web.append(generated)
    st.session_state.web_citations.append(web_response.citations or [])
    st.session_state.web_documents.append(web_response.documents or [])

if st.session_state.generated_tt:
    with tt_tab:
        tt_chat, tt_refs = st.columns(2)

        with tt_chat:
            for i in range(len(st.session_state.generated_tt)):
                message(st.session_state.past[i], is_user=True, allow_html=True, key=str(uuid.uuid1()))
                message(st.session_state.generated_tt[i], allow_html=True, key=str(uuid.uuid1()))

        with tt_refs:
            for i in range(len(st.session_state.generated_tt)):
                for j in range(len(st.session_state.tt_documents[i])):
                    doc = st.session_state.tt_documents[i][j]
                    ref_anchor = f"<a id='tt_ref_{i + 1}_{j + 1}'>Reference {i + 1}.{j + 1}</a>"
                    snippet = doc['snippet'].replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
                    message(f"{ref_anchor}:\n{snippet}", is_user=True, allow_html=True, avatar_style="shapes", key=str(uuid.uuid1()))

if st.session_state.generated_web:
    with web_tab:
        web_chat, web_refs = st.columns(2)

        with web_chat:
            for i in range(len(st.session_state.generated_web)):
                message(st.session_state.past[i], is_user=True, allow_html=True, key=str(uuid.uuid1()))
                message(st.session_state.generated_web[i], allow_html=True, key=str(uuid.uuid1()))

        with web_refs:
            for i in range(len(st.session_state.generated_web)):
                for j in range(len(st.session_state.web_documents[i])):
                    doc = st.session_state.web_documents[i][j]
                    ref_anchor = f"<a id='web_ref_{i + 1}_{j + 1}'>Reference {i + 1}.{j + 1}</a>"
                    snippet = doc['snippet'].replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
                    message(f"{ref_anchor}:\n{snippet}", is_user=True, allow_html=True, avatar_style="shapes", key=str(uuid.uuid1()))
