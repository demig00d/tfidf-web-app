from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import numpy as np
from typing import List, Dict
import operator

tfidf_vectorizer = TfidfVectorizer(token_pattern=r"\b[^\d\W]+\b")


def round_if_close(x: np.float64, decimals: int):
    round_x = np.round(x, decimals)
    return round_x if np.isclose(x, round_x) else x


def tfidf(documents: List[str]) -> List[Dict[str, str | np.float64]]:
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
        {
            "слово": word,
            "tf":    round_if_close(tf_score, decimals=2),
            "idf":   round_if_close(idf_score, decimals=2)
        }
        for word, tf_score, idf_score in sorted_scores
    ]

    return tf_idf_scores
