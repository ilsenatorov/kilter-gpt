import torch

from kiltergpt.data.tokenizer import Tokenizer  # Adjust the import path as necessary


def test_init():
    """Tests that the tokenizer initializes correctly."""
    tokenizer = Tokenizer()
    assert isinstance(tokenizer, Tokenizer)
    assert len(tokenizer.hold_token_ids) == 476  # original kilterboard has 527 holds
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


def test_tokenizer_encode():
    tokenizer = Tokenizer()
    encoded = tokenizer.encode("p1100r12", 10, "7a")
    # 6 tokens: [BOS], [angle], [grade], p1100, r12, [EOS]
    assert encoded.size(0) == 6


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


def test_tokenizer_onehot():
    tokenizer = Tokenizer()
    frames = "p1100r12p1200r13p1300r14"
    encoded = tokenizer.encode(frames, 10, "7a")
    onehot_from_str = tokenizer.onehot(frames)
    onehot_from_encoded = tokenizer.onehot(encoded)
    assert (onehot_from_str == onehot_from_encoded).all()
    assert onehot_from_str.sum() == 3
