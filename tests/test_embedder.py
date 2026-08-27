import pytest

from app.embedder import EmbeddingClient


@pytest.fixture(scope="module")
def client():
    return EmbeddingClient()


def test_embed_returns_one_vector_per_text(client):
    vectors = client.embed(["hello world", "another sentence"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


def test_embed_empty_list_returns_empty(client):
    assert client.embed([]) == []


def test_embed_similar_texts_are_closer_than_dissimilar(client):
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    v_diabetes_a, v_diabetes_b, v_unrelated = client.embed(
        [
            "Metformin is a first-line treatment for type 2 diabetes.",
            "Type 2 diabetes is commonly treated with metformin first.",
            "The mayor of Fixtureville was appointed in 2020.",
        ]
    )

    assert cosine(v_diabetes_a, v_diabetes_b) > cosine(v_diabetes_a, v_unrelated)
