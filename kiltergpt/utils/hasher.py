import hashlib
import json
import random


class Hasher:
    """Given a frames string, this class generates a random adjective-noun pair based on the hash of the string."""

    def __init__(self, adjectives: list, nouns: list, sep: str = " "):
        self.adjectives = adjectives
        self.nouns = nouns
        self.sep = sep

    @staticmethod
    def sort_holds(inp: str) -> str:
        """Specific to kilter frames."""
        return "".join(sorted([f"p{x}" for x in inp.split("p")[1:]]))

    def encode(self, frames: str) -> str:
        """From frames to code"""
        frames = self.sort_holds(frames)
        hash_object = hashlib.sha256(frames.encode()).hexdigest()
        random.seed(hash_object)
        adj = random.choice(self.adjectives)
        noun = random.choice(self.nouns)
        return f"{adj}{self.sep}{noun}"

    @staticmethod
    def from_json(json_path: str = "data/words.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        return Hasher(data["adjectives"], data["nouns"])
