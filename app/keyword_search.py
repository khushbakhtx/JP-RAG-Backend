from keybert import KeyBERT

keybert_model = KeyBERT()

def extract_keywords_from_text(text, top_n=5):
    """
    Extract keywords from the text using KeyBERT.
    """
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n)
    return [keyword[0] for keyword in keywords]

def search_using_keywords(document_store, user_query, k_candidates=2):
    """
    Perform keyword-based search for relevant chunks.
    """
    # Extract keywords from the user query
    keywords = extract_keywords_from_text(user_query)

    relevant_docs = []
    
    # Iterate over the document store
    for doc_id, doc in document_store.items():
        doc_keywords = extract_keywords_from_text(doc["content"])
        if any(keyword in doc_keywords for keyword in keywords):
            relevant_docs.append(doc)

    return relevant_docs[:k_candidates]
