import json

from ..data import Tokenizer


class Hasher(Tokenizer):
    """Given a frames string, this class generates a random adjective-noun pair based on the hash of the string."""

    def __init__(self, adjectives: list, nouns: list, sep: str = " "):
        super().__init__()
        self.adjectives = adjectives
        self.nouns = nouns
        self.sep = sep

    @staticmethod
    def sort_holds(inp: str) -> str:
        """Specific to kilter frames."""
        return "".join(sorted([f"p{x}" for x in inp.split("p")[1:]]))

    def encrypt(self, frames: str) -> str:
        """From frames to code"""
        inp = self.sort_holds(frames)

    def decrypt(self, code: str) -> str:
        """From code to frames"""
        adjective, noun = code.split(self.sep)

    @staticmethod
    def from_json(json_path: str = "data/words.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        return Hasher(data["adjectives"], data["nouns"])
