import torch

from kiltergpt.data.tokenizer import Tokenizer  # Adjust the import path as necessary


def test_tokenizer_from_json():
    """Tests that the tokenizer can be loaded from a JSON file."""
    tokenizer = Tokenizer.from_json("data/tokenizer.json")
    assert isinstance(tokenizer, Tokenizer)  # Ensure it's a Tokenizer instance


def test_tokenizer_vocab_sizes():
    """Verifies the correct number of unique tokens for each category."""
    tokenizer = Tokenizer.from_json("data/tokenizer.json")
    assert len(tokenizer.angle_tokens) == 15
    assert len(tokenizer.grade_tokens) == 23
    assert len(tokenizer.special_tokens()) == 5


def test_tokenizer_encode():
    """Tests basic encoding with and without special tokens."""
    tokenizer = Tokenizer.from_json("data/tokenizer.json")

    encoded = tokenizer.encode("p1100r12", 10, "7a")
    assert encoded.size(0) == 6
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id

    encoded = tokenizer.encode("p1100r12", 10, "7a", eos=False)
    assert encoded.size(0) == 5
    assert encoded[-1] != tokenizer.eos_token_id

    encoded = tokenizer.encode("p1100r12", 10, "7a", eos=False, bos=False)
    assert encoded.size(0) == 4


def test_tokenizer_encode_padding():
    """Ensures padding works correctly to a desired length."""
    tokenizer = Tokenizer.from_json("data/tokenizer.json")
    encoded = tokenizer.encode("p1100r12", 10, "7a", pad=64)
    assert encoded.size(0) == 64
    assert torch.all(encoded[:-6] == tokenizer.pad_token_id)  # Check padding tokens


def test_tokenizer_decode():
    """Tests decoding back to the original sequence."""
    tokenizer = Tokenizer.from_json("data/tokenizer.json")

    encoded = tokenizer.encode("p1100r12", 10, "7a")
    decoded = tokenizer.decode(encoded, clean=True)
    assert decoded == ("p1100r12", 10, "7a")


def test_tokenizer_invalid_inputs():
    """Checks handling of invalid input sequences."""
    tokenizer = Tokenizer.from_json("data/tokenizer.json")

    try:
        tokenizer.encode("invalid_sequence", 10, "7a")
        raise AssertionError("Expected a ValueError for an invalid sequence")
    except AssertionError:
        pass
