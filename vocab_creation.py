import json
import nltk

def vocab_creation(corpus_path):
    """Creates a set of the non-stopword tokens of corpus."""

    corpus_meaningful_tokens = []
    with open(corpus_path) as f:
        for line in f:
            # tokenize line:
            line_dict = json.loads(line)
            abstract = line_dict["abstract"]
            abstract_tokens = [nltk.tokenize.word_tokenize(sent) for sent in abstract]
            # adding important tokens to vocabulary:
            stopwords = nltk.corpus.stopwords.words("english")
            for sent in abstract_tokens:
                for word in sent:
                    if word.lower() not in stopwords and word.lower().isalpha():
                        corpus_meaningful_tokens.append(word.lower())
            
    corpus_vocab = set(corpus_meaningful_tokens)

    return corpus_vocab
