# src/rag/retriever/reranker.py
# --- TO DO : Improve Reranker class, instantiate with config parameter, change top_k thanks to config parameter --
from sentence_transformers import CrossEncoder

class ReRanker:
    """
    Cross-encoder-based reranker for refining retrieved document rankings.
    Re-scores query–document pairs using a transformer model.
    """

    # Initialize the reranker with a pretrained cross-encoder model.
    # The model predicts a semantic relevance score for each query–document pair.
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name (str): Name of the cross-encoder model from the SentenceTransformers library.
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    # Rerank retrieved results using cross-encoder scores.
    # Computes pairwise relevance scores and sorts results by descending score.
    def rerank(self, query, results, top_k=None):
        """
        Re-rank retrieved results by predicting semantic similarity between query and documents.

        Args:
            query (str): The input query string.
            results (list): List of retrieved documents with text fields.
            top_k (int, optional): Number of top documents to return after reranking.

        Returns:
            list: Re-ranked list of documents with added 'rerank_score' field.
        """
        test_top_k =5
        # Prepare (query, document) pairs for batch scoring with the cross-encoder.
        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        # Sort documents by predicted relevance score in descending order.
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        if test_top_k:
            results = results[:test_top_k]
        return results
