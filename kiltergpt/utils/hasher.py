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


# import json
# from kiltergpt.data import Tokenizer


# class Hasher(Tokenizer):
#     """Given a frames string, this class generates a random adjective-adjective-noun-noun pair based on the hash of the string."""

#     def __init__(self, adjectives: list, nouns: list, sep: str = " "):
#         self.adjectives = adjectives
#         self.nouns = nouns
#         self.sep = sep
#         mapping = []
#         for i in Tokenizer.hold_tokens():
#             for j in Tokenizer.color_tokens():
#                 mapping.append(f"{i}{j}")
#         self.mapping = {token: i for i, token in enumerate(mapping)}
#         self.reverse_mapping = {v: k for k, v in self.mapping.items()}

#     @staticmethod
#     def sort_holds(inp: str) -> str:
#         """Specific to kilter frames."""
#         return "".join(sorted([f"p{x}" for x in inp.split("p")[1:]]))

#     def encrypt(self, frames: str) -> str:
#         """From frames to code"""
#         frames = self.sort_holds(frames)
#         combined_index = 0
#         base = len(self.mapping)
#         for token in frames.split("p")[1:]:
#             combined_index = combined_index * base + self.mapping[f"p{token}"]

#         adjective_index_1 = combined_index % len(self.adjectives)
#         adjective_index_2 = (combined_index // len(self.adjectives)) % len(self.adjectives)
#         noun_index_1 = (combined_index // (len(self.adjectives) ** 2)) % len(self.nouns)
#         noun_index_2 = (combined_index // (len(self.adjectives) ** 2 * len(self.nouns))) % len(self.nouns)

#         adjective_1 = self.adjectives[adjective_index_1]
#         adjective_2 = self.adjectives[adjective_index_2]
#         noun_1 = self.nouns[noun_index_1]
#         noun_2 = self.nouns[noun_index_2]

#         return f"{adjective_1}{self.sep}{adjective_2}{self.sep}{noun_1}{self.sep}{noun_2}"

#     def decrypt(self, code: str) -> str:
#         """From code to frames"""
#         adjective_1, adjective_2, noun_1, noun_2 = code.split(self.sep)
#         adjective_index_1 = self.adjectives.index(adjective_1)
#         adjective_index_2 = self.adjectives.index(adjective_2)
#         noun_index_1 = self.nouns.index(noun_1)
#         noun_index_2 = self.nouns.index(noun_2)

#         combined_index = (
#             adjective_index_1
#             + adjective_index_2 * len(self.adjectives)
#             + noun_index_1 * len(self.adjectives) ** 2
#             + noun_index_2 * len(self.adjectives) ** 2 * len(self.nouns)
#         )

#         base = len(self.mapping)
#         token_list = []
#         while combined_index > 0:
#             token_index = combined_index % base
#             token_list.append(self.reverse_mapping[token_index])
#             combined_index //= base

#         token_list.reverse()
#         frames = "".join(token_list)
#         return frames

#     @staticmethod
#     def from_json(json_path: str = "data/words.json"):
#         with open(json_path, "r") as f:
#             data = json.load(f)
#         return Hasher(data["adjectives"], data["nouns"])
