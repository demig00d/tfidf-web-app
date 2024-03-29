from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import numpy as np
from typing import List, Dict
import operator

tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b[^\d\W]+\b')


def tfidf(documents: List[str]) -> List[Dict[str, str]]:
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    feature_names = tfidf_vectorizer.get_feature_names_out()

    idf_scores = tfidf_vectorizer.idf_

    tf_scores = (
        tfidf_matrix.toarray() / tfidf_matrix.toarray().sum(axis=1)[:, np.newaxis]
    )

    sorted_scores = sorted(
        zip(feature_names, tf_scores.mean(axis=0), idf_scores),
        key=operator.itemgetter(2),
        reverse=True,
    )[:50]

    tf_idf_scores = [
        {"слово": word, "tf": tf_score, "idf": idf_score}
        for word, tf_score, idf_score in sorted_scores
    ]

    return tf_idf_scores
