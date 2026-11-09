import numpy as np
import pytest

from finqa.train.model import BiEncoder

NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCES = ["Should I pay off my mortgage early?",
             "401k and IRA contribution limits for this year",
             "What is a dividend?",
             "word " * 400]  # long: exercises truncation to 256 tokens


def test_matches_sentence_transformers_embeddings():
    st = pytest.importorskip("sentence_transformers")
    ours = BiEncoder(NAME).encode(SENTENCES)
    ref = st.SentenceTransformer(NAME, device="cpu").encode(SENTENCES, normalize_embeddings=True)
    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, atol=1e-4)


def test_embeddings_are_unit_length_and_order_is_preserved():
    m = BiEncoder(NAME)
    emb = m.encode(SENTENCES, batch_size=2)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert np.allclose(emb[1], m.encode([SENTENCES[1]])[0], atol=1e-5)
