from kiltergpt.data.tokenizer import Tokenizer  # Adjust the import path as necessary


def test_tokenizer_base():
    tokenizer = Tokenizer.from_json("data/tokenizer.json")
    assert len(tokenizer.angle_tokens) == 15  # 15 unique angles
    assert len(tokenizer.grade_tokens) == 23  # 23 unique grades
    assert len(tokenizer.special_tokens()) == 5  # 5 special tokens


def test_tokenizer_encode():
    tokenizer = Tokenizer.from_json("data/tokenizer.json")
    tokens = tokenizer.encode("p1100r12", 10, "7a")
    print(tokens)
