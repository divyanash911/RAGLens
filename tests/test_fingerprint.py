"""Tests for config_fingerprint determinism and sensitivity."""

from raglens.fingerprint import config_fingerprint


def test_deterministic_and_order_independent():
    a = config_fingerprint({"embedding_model": "bge", "chunking": {"size": 256}})
    b = config_fingerprint({"chunking": {"size": 256}, "embedding_model": "bge"})
    assert a == b


def test_changes_with_config():
    a = config_fingerprint({"embedding_model": "bge"})
    b = config_fingerprint({"embedding_model": "e5"})
    assert a != b


def test_length_param():
    assert len(config_fingerprint({"x": 1}, length=8)) == 8


def test_empty_config():
    assert isinstance(config_fingerprint({}), str)
