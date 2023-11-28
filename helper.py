from cohere.client import Chat


def insert_substring(source_str: str, insert_str: str, pos: int) -> str:
    return source_str[:pos] + insert_str + source_str[pos:]


def mark_citations(kind: str, index: int, response: Chat) -> str:
    doc_map = dict()
    for idx, doc in enumerate(response.documents or []):
        if doc["id"] not in doc_map:
            doc_map[doc["id"]] = "#" + f"{kind}_ref_{index + 1}_{idx + 1}"

    txt = response.text or "N/A"
    for cit in (response.citations or [])[::-1]:
        if not cit["document_ids"]:
            continue

        doc_anchor = doc_map[cit["document_ids"][0]]
        cit_idxs = doc_anchor.split("_")
        cit_sup = f"{cit_idxs[-2]}.{cit_idxs[-1]}"
        txt = insert_substring(txt, f"<sup>{cit_sup}</sup></a>", cit["end"])
        txt = insert_substring(txt, f"<a href='{doc_anchor}' target='_self'>", cit["start"])

    return txt
