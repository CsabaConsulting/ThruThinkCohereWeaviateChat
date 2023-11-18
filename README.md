# ThruThink® Support Chat Agent utilizing RAG Fusion, powered by [Cohere](https://cohere.com/) & [Weaviate](https://weaviate.io/)

The main business goal is to develop a support chat agent for the investment projection web application [ThruThink®](https://thruthink.com).
* ThruThink® is a business budgeting on-line app to create professional budgets and forecasts
* A product of literally decades of experience and careful thought, and thousands of calculations
* Thru-hiking, or through-hiking, is the act of hiking an established long-distance trail end-to-end continuously
* There are no dedicated personnel for support chat agent roles, it had a “classic” chat agent integration in the past
* An LLM and RAG (Retrieval Augmented Generation) powered chat agent could be invaluable, given that
  1. It stays relatively grounded
  2. It Won’t hallucinate* wildly

## Desired abilities:
* Main goal: answer ThruThink® software specific questions such as: "In ThruThink can I make adjustments on the Cash Flow Control page?"
* Nice to have: answer more generic questions such as: "How much inventory should I have?"

The bulk of the knowledge base consists of close to 190 help topics also divided into a few dozen categories.
That's more than nothing, however users can ask such a wide variety of questions that chunking these documents may not provide a nice ground for a good vector match in the embedding space. To increase the performance of the chat agent I employ several techniques.

## Achievements:
1. With a synthetic data generation I enriched the knowledge base. I coined this QBRAG (QnA Boosted RAG) because I'm using the same QnA data I already generated and curated for potential fine tuning purposes. The same dataset can be used to enrich the vector indexed knowledge as well.
2. The highlight of my submission is RAG Fusion ([see article](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)).
3. I utilize [Weaviate](https://weaviate.io/) for vector storage, embedding, matching and retrieval. I use [Cohere](https://cohere.com/) for multiple language models: fine tuned chat model and also `co.chat` with a web connector advanced feature in different stages of the chain.
4. I also use [LangChain](https://www.langchain.com/) to construct some stages of the chain.
5. The front-end is powered by and hosted on [Streamlit](https://streamlit.io/), I highly customized the view which also features linked references.
6. After the fusion and re-ranking I provide the user with both results from a more traditional RAG grounded `co.chat` call and also a web-connector powered other call (that is also augmented to provide guidance) to show both information so the user can get the best of both worlds.
7. Since I have to control several stages of the chain for the fusion, I was not able to use such high level [LangChain](https://www.langchain.com/) constructs as `ConversationalRetrievalChain` or `RetrievalQA`, so `co.chat`'s ability to handle the conversation for me (via `conversation_id`) made my job much easier than I'd have to work for history / memory functionality and other building blocks.

## RAG Fusion:
1. Since users might ask questions which are not present in the QnA in the particular form but still covered by the knowledge, the application first generates variations of the user's query with the help of a fine tuned Cohere model. The hope is that some of these variations may match closer to some QnA or help data chunks.
2. Then document retrieval happens for all of the query variations.
3. There's a reciprocal rank fusion which concludes a fused list of documents across all the variations.
4. We'll take the top k of those documents and perform final two RAG calls which supply the displayed data.

## Other achievements:
* I added the capability to the https://github.com/nestordemeure/question_extractor open source project to continue a long running interrupted QnA generation session (it can take many hours with default rate limits on both OpenAI or AnyScale). See my open repository https://github.com/CsabaConsulting/question_extractor
* I developed a script which can convert the standard format fine tuning `jsonl` into a set of markdown files (QnA grouped by help topics) for Weaviate indexing. The script breaks apart the `questions.jsonl` into the files based on which file the questions originally sourced from. This is needed for the QBRAG synthetic data indexing. For the conversion see the https://github.com/CsabaConsulting/question_extractor/blob/main/augment_prep.py script.
* I added capability for my script to produce a fine tuning file for the Cohere LLM model (see https://github.com/CsabaConsulting/question_extractor/blob/main/fine_tune_prep.py).

Kindly look at the development and experimentation IPython notebooks and scripts in the https://github.com/CsabaConsulting/Cohere repository. These were used to establish the Weaviate schema for ingestion / indexing / retrieval, and also testing retrieval and building up the parts for the RAG Fusion.

## Future plans:
1. Decrease runtime by running the variation document retrievals in parallel, this is a [Streamlit](https://streamlit.io/) specific tech challenge with asyncio / await.
2. Decrease runtime by running the final two co.chat RAG calls in parallel, this is a [Streamlit](https://streamlit.io/) specific tech challenge with asyncio / await.
3. Make the citation linking nicer and other UI enhancements.
4. Measure how much the RAG Fusion (if at all) improves answer quality. Measure the trade-off factoring in extra latency and potential token usage increase which also means cost increase.
5. Integrate the agent into ThruThink which uses ASP.NET MVC / C# / Azure technology stack, but not open source. In that final deployment I'll be able to open up referred help topics using the meta-data I get back as part of the query results document metadata.
6. Add filter against harmful content, for example using [Google PaLM2's safety attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#attribute-descriptions).
