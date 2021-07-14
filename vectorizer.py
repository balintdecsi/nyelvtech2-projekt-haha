import json
from sklearn.feature_extraction.text import TfidfVectorizer

def content_concatenator(jsonl_path, field_name):
    """Concatenates all individual sentences of the field of interest in .jsonl into one list."""

    content_sents = []
    with open(jsonl_path) as f:
        for line in f:
            line_dict = json.loads(line)
            content = line_dict[field_name]
            if field_name == "abstract":
                content_sents += [cont_sent for cont_sent in content]
            else:
                content_sents = content

    return content_sents
            

def content2vec(content_path, content_name, *args):
    """Vectorizes all of the sentences of the content (claims or corpus(titles excluded)) into TF-IDF features."""
    
    content_sents = content_concatenator(content_path, content_name)
    if args:
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", vocabulary=args[0])
    else:
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    content_vec = tfidf_vectorizer.fit_transform(content_sents)

    return content_vec.A, tfidf_vectorizer.vocabulary
