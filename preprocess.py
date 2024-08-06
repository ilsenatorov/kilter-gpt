import sqlite3
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from kiltergpt.utils import KilterPolice

parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data", help="Directory to save data, has to contain db.sqlite3")
parser.add_argument("--min_ascents", type=int, default=1, help="Minimum number of ascents")
parser.add_argument("--min_quality", type=int, default=2, help="Minimum quality")
parser.add_argument("--min_holds", type=int, default=4, help="Minimum number of holds")
parser.add_argument("--max_holds", type=int, default=28, help="Maximum number of holds")
args = parser.parse_args()


data_path = Path("data")
data_path.mkdir(exist_ok=True)
(data_path / "raw").mkdir(exist_ok=True)
(data_path / "processed").mkdir(exist_ok=True)

# load everything from sql
# data/db.sqlite3 you can get from the latest kilterboard apk
conn = sqlite3.connect("data/db.sqlite3")
climbs = pd.read_sql_query("SELECT * FROM climbs", conn)
grades = pd.read_sql_query("SELECT * FROM difficulty_grades", conn)
stats = pd.read_sql_query("SELECT * FROM climb_stats", conn)
holds = pd.read_sql_query("SELECT * FROM holes", conn)
placements = pd.read_sql_query("SELECT * FROM placements", conn)
holds = pd.merge(placements, holds, left_on="hole_id", right_on="id")
holds.set_index("id_x", inplace=True)

# merge and rename
df = pd.merge(climbs.drop("angle", axis=1), stats, left_on="uuid", right_on="climb_uuid")
df["average_grade"] = df["difficulty_average"].apply(lambda x: grades.loc[int(x) + 1, "boulder_name"])
df["font_grade"] = df["average_grade"].apply(lambda x: x.split("/")[0])
df["v_grade"] = df["average_grade"].apply(lambda x: x.split("/")[1])


print(df.shape)
df = df[df["frames_count"] == 1]
print(df.shape)
df = df[df["is_listed"] == 1]
print(df.shape)
df = df[df["layout_id"] == 1]
print(df.shape)
df = df[df["quality_average"] >= args.min_quality]
print(df.shape)
df = df[df["ascensionist_count"] >= args.min_ascents].reset_index()
print(df.shape)

holds = holds[holds["layout_id"] == 1]  # only original boards
holds = holds[holds.index.to_series() < 3000]


kp = KilterPolice(set(holds.index), n_total_holds=(args.min_holds, args.max_holds))
df["valid"] = df["frames"].apply(kp.check)
df = df[df["valid"]]

df.to_csv("data/processed/all_climbs.csv")
holds.to_csv("data/processed/holds.csv")
grades.to_csv("data/processed/grades.csv")

# split into train, val and test
df = df.sample(frac=1)
train = df.iloc[: int(0.8 * len(df))]
val = df.iloc[int(0.8 * len(df)) : int(0.9 * len(df))]
test = df.iloc[int(0.9 * len(df)) :]
train.to_csv("data/processed/train.csv")
val.to_csv("data/processed/val.csv")
test.to_csv("data/processed/test.csv")


### for plotter uses
# holds['img_x'] = (7.5 * holds['x']).astype(int)
# holds['img_y'] = (-7.5 * holds['y'] + 1171).astype(int)
# holds[['img_x', 'img_y']].to_csv("figs/image_coords.csv")
# !wget https://raw.githubusercontent.com/Declan-Stockdale-Garbutt/KilterBoard_climb_generator/main/data/full_board_commercial.png
# !mv full_board_commercial.png figs/
# !convert figs/full_board_commercial.png -define png:color-type=2 figs/full_board_commercial.png
