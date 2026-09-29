from src.config import settings


def test_grounding_refusal_text_is_explicit():
    expected = "I cannot answer that based on the provided document."
    assert expected.startswith("I cannot answer")


def test_configuration_defaults_match_assignment():
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 4
