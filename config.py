boundaries = {
    "variation_count_min": 2,
    "variation_count_max": 10,
    "document_k_min": 3,
    "document_k_max": 20,
    "document_k": 15,
    "top_k_augment_doc_min": 2,
    "top_k_augment_doc_max": 10,
    "temperature_min": 0.0,
    "temperature_max": 1.0,
}
defaults = {
    "variation_count": 4,
    "index_name": "Help",
    "text_key": "text",
    "document_k": 15,
    "rerank_k": 60,
    "top_k_augment_doc": 5,
    "temperature": 0.0,
    "cohere_variation_model": "d2cd24fd-67fd-4610-ab84-a37f9e635313-ft",
    "cohere_fusion_model": "command-nightly",
    "max_retries": 5,
}
