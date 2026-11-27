from meme_organizer.retrieval.text.dense_text_retriever import DenseTextRetriever


def test_dense_text_retriever_init() -> None:
    retriever = DenseTextRetriever(
        model_name="rubert-tiny2",
        hf_model_id="cointegrated/rubert-tiny2",
    )
    assert retriever.model_name == "rubert-tiny2"
    assert retriever.hf_model_id == "cointegrated/rubert-tiny2"
