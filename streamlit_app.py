import cohere
import os
import streamlit as st
import uuid
import weaviate

from cohere.client import Chat
from config import defaults, boundaries
from helper import mark_citations
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere
from langchain.vectorstores import Weaviate
from rag_fusion import (
  extract_query_variations,
  final_rag_operations,
  generate_variations,
  rerank_and_fuse_documents,
  retrieve_documents_for_query_variations,
)
from streamlit_chat import message


title = "ThruThink Support"
st.set_page_config(
    page_title=title,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    "[![ThruThink logo](app/static/thruthink_logo.png)](https://thruthink.com)" +
    "[![5 minute pitch video](app/static/yt_logo.png)](https://youtu.be/SzJB-5rUmDg?t=227)" +
    "[![lablab project](app/static/link_symbol.png)](https://lablab.ai/event/cohere-coral-hackathon/thruthink/rag-fusion-with-cohere-and-weaviate)"
)


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


def generate_response_with_rag_fusion(query: str) -> tuple[Chat, Chat]:
    # Step 1: Generate query variations
    with st.spinner("Generating variations..."):
        query_variations = generate_variations(query, variation_count, llm)
        queries = extract_query_variations(query, query_variations, variation_count)

    # Step 2: Retrieve documents for each query variation
    with st.spinner("Retrieving for variations and reranking..."):
        document_sets = retrieve_documents_for_query_variations(queries, vectorstore, document_k)

        # Step 3: Rerank the document sets with reciprocal rank fusion
        reranked_results = rerank_and_fuse_documents(document_sets, rerank_k)

    # Step 4: Prepare and executing final RAG calls (a grounded and a web)
    with st.spinner("Executing twin queries..."):
        tt_response, web_response = final_rag_operations(
            query,
            reranked_results,
            top_k_augment_doc,
            cohere_fusion_model,
            temperature,
            conversation_id,
            co
        )

    return tt_response, web_response


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
