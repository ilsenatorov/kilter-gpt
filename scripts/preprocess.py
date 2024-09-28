import sqlite3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas as pd

from kiltergpt.data.tokenizer import Tokenizer
from kiltergpt.utils import KilterPolice

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--sqlite_path", type=Path, default="data/db.sqlite3", help="Path to sqlite3 file")
parser.add_argument("--out_dir", type=Path, default="data/processed", help="Directory to save data")
parser.add_argument("--min_ascents", type=int, default=1, help="Minimum number of ascents")
parser.add_argument("--min_quality", type=float, default=2, help="Minimum quality")
parser.add_argument("--start_holds", type=int, nargs=2, default=(0, 2), help="Number of start holds")
parser.add_argument("--finish_holds", type=int, nargs=2, default=(0, 2), help="Number of finish holds")
parser.add_argument("--hand_holds", type=int, nargs=2, default=(0, 999), help="Number of hand holds")
parser.add_argument("--foot_holds", type=int, nargs=2, default=(0, 999), help="Number of foot holds")
parser.add_argument("--total_holds", type=int, nargs=2, default=(4, 28), help="Number of total holds")
parser.add_argument("--data_split", type=float, nargs=3, default=[0.9, 0.09, 0.01], help="How to split the data")
args = parser.parse_args()

assert sum(args.data_split) == 1, "Data split fractions must sum to 1."
assert len(args.data_split) == 3, "Must have 3 splits for train, val and test."
args.out_dir.mkdir(exist_ok=True, parents=False)

# load everything from sql
# data/db.sqlite3 you can get from the latest kilterboard apk
print("Loading data from sqlite3")
conn = sqlite3.connect(args.sqlite_path)
climbs = pd.read_sql_query("SELECT * FROM climbs", conn)
grades = pd.read_sql_query("SELECT * FROM difficulty_grades", conn).set_index("difficulty")
stats = pd.read_sql_query("SELECT * FROM climb_stats", conn)
holds = pd.read_sql_query("SELECT * FROM holes", conn)
placements = pd.read_sql_query("SELECT * FROM placements", conn)
holds = pd.merge(placements, holds, left_on="hole_id", right_on="id")
holds.set_index("id_x", inplace=True)

# merge and rename
df = pd.merge(climbs.drop("angle", axis=1), stats, left_on="uuid", right_on="climb_uuid")
df["average_grade"] = df["difficulty_average"].apply(lambda x: grades.loc[int(round(x, 0)), "boulder_name"])
df["font_grade"] = df["average_grade"].apply(lambda x: x.split("/")[0])
df["v_grade"] = df["average_grade"].apply(lambda x: x.split("/")[1])


print(f"Total climbs:\n{df.shape[0]}")
df = df[df["frames_count"] == 1]
print(f"Removing route climbs:\n{df.shape[0]}")
df = df[df["is_listed"] == 1]
print(f"Removing unlisted climbs:\n{df.shape[0]}")
df = df[df["layout_id"] == 1]
print(f"Removing non-original boards:\n{df.shape[0]}")
df = df[df["quality_average"] >= args.min_quality]
print(f"Removing low quality climbs (quality threshold={args.min_quality}):\n{df.shape[0]}")
df = df[df["ascensionist_count"] >= args.min_ascents].reset_index()
print(f"Removing climbs with less than {args.min_ascents} ascents:\n{df.shape[0]}")

holds = holds[holds["layout_id"] == 1]  # only original boards
holds = holds[holds.index.to_series() < 1800]


kp = KilterPolice(Tokenizer(), args.start_holds, args.finish_holds, args.foot_holds, args.hand_holds, args.total_holds)
df["valid"] = df["frames"].apply(kp.check)
df[~df["valid"]].to_csv(args.out_dir / "invalid_climbs.csv")
df = df[df["valid"]]
print(f"Removing invalid climbs, they are saved to 'invalid_climbs.csv':\n{df.shape[0]}")

holds.to_csv(args.out_dir / "holds.csv")
grades.to_csv(args.out_dir / "grades.csv")

print("Splitting the data")
train_frac, val_frac, test_frac = args.data_split
# split into train, val and test
df = df.sample(frac=1)  # shuffle
train = df.iloc[: int(train_frac * len(df))]
val = df.iloc[int(train_frac * len(df)) : int((train_frac + val_frac) * len(df))]
test = df.iloc[int((train_frac + val_frac) * len(df)) :]
train.to_csv(args.out_dir / "train.csv")
val.to_csv(args.out_dir / "val.csv")
test.to_csv(args.out_dir / "test.csv")


### for plotter uses
# print("Creating image coordinates")
# holds['img_x'] = (7.5 * holds['x']).astype(int)
# holds['img_y'] = (-7.5 * holds['y'] + 1171).astype(int)
# holds[['img_x', 'img_y']].to_csv("figs/image_coords.csv")
# !wget https://raw.githubusercontent.com/Declan-Stockdale-Garbutt/KilterBoard_climb_generator/main/data/full_board_commercial.png
# !mv full_board_commercial.png figs/
# !convert figs/full_board_commercial.png -define png:color-type=2 figs/full_board_commercial.png
