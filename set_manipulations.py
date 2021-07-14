import numpy as np

def X_creator(claim_array, corpus_array):
    """Creates an X set for machine learning from claim and corpus sentence arrays."""
    
    assert claim_array.shape[1] == corpus_array.shape[1], "No. of features do not equal!"
    no_of_corpus_sents = corpus_array.shape[0]
    no_of_claim_sents = claim_array[0]
    no_of_features = claim_array.shape[1]
    
    X_claim = np.array((1,no_of_features))
    for i in range(0, no_of_claim_sents):
        X_claim = np.vstack((X_claim, np.broadcast_to(claim_array[i][:],(no_of_corpus_sents,no_of_features))))

    X_corpus = np.broadcast_to(corpus_array, (no_of_corpus_sents*no_of_claim_sents,no_of_features))

    X = np.hstack(X_claim, X_corpus)

    return X