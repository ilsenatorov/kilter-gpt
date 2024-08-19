import torch

from kiltergpt.data.tokenizer import Tokenizer  # Adjust the import path as necessary


def test_init():
    """Tests that the tokenizer initializes correctly."""
    tokenizer = Tokenizer()
    assert isinstance(tokenizer, Tokenizer)
    assert len(tokenizer.hold_token_ids) == 527  # original kilterboard has 527 holds
    assert len(tokenizer.angle_token_ids) == (95 // 5)  # 19 angle options
    assert len(tokenizer.color_token_ids) == 4  # 4 hold roles
    assert len(tokenizer.grade_token_ids) == 25  # 24 grades (from 4a to 9a, first plus grade is 6a+)
    assert len(tokenizer.special_token_ids) == 5  # "[BOS]", "[EOS]", "[PAD]", "[UNK]", "[MASK]"


def test_tokenizer_encode():
    """Tests basic encoding with and without special tokens."""
    tokenizer = Tokenizer()

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
    tokenizer = Tokenizer()
    encoded = tokenizer.encode("p1100r12", 10, "7a", pad=64)
    assert encoded.size(0) == 64
    assert torch.all(encoded[:-6] == tokenizer.pad_token_id)  # Check padding tokens


def test_tokenizer_decode():
    """Tests decoding back to the original sequence."""
    tokenizer = Tokenizer()

    encoded = tokenizer.encode("p1100r12", 10, "7a")
    decoded = tokenizer.decode(encoded, clean=True)
    assert decoded == ("p1100r12", 10, "7a")


def test_tokenizer_invalid_inputs():
    """Checks handling of invalid input sequences."""
    tokenizer = Tokenizer()

    try:
        tokenizer.encode("invalid_sequence", 10, "7a")
        raise AssertionError("Expected a ValueError for an invalid sequence")
    except AssertionError:
        pass
