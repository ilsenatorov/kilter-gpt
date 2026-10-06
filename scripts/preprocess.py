import sqlite3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas as pd

from kiltergpt.data.datasets import deduplicate_climbs, post_archive_holdout, restore_legacy_stats, split_by_climb
from kiltergpt.data.tokenizer import Tokenizer
from kiltergpt.utils import KilterPolice

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--sqlite_path", type=Path, default="data/db.sqlite3", help="Path to sqlite3 file")
parser.add_argument(
    "--metadata_path",
    type=Path,
    default="data/db.sqlite3",
    help="Legacy database for board metadata when using a snapshot",
)
parser.add_argument("--out_dir", type=Path, default="data/processed", help="Directory to save data")
parser.add_argument("--min_ascents", type=int, default=1, help="Minimum number of ascents")
parser.add_argument("--min_quality", type=float, default=2, help="Minimum quality")
parser.add_argument(
    "--broad", action="store_true", help="Keep zero-ascent climbs; exclude positively climbed routes below min_quality"
)
parser.add_argument("--start_holds", type=int, nargs=2, default=(1, 2), help="Number of start holds")
parser.add_argument("--finish_holds", type=int, nargs=2, default=(1, 2), help="Number of finish holds")
parser.add_argument("--hand_holds", type=int, nargs=2, default=(0, 999), help="Number of hand holds")
parser.add_argument("--foot_holds", type=int, nargs=2, default=(0, 999), help="Number of foot holds")
parser.add_argument("--total_holds", type=int, nargs=2, default=(4, 28), help="Number of total holds")
args = parser.parse_args()
args.out_dir.mkdir(exist_ok=True, parents=False)

# load everything from sql
# data/db.sqlite3 you can get from the latest kilterboard apk
print("Loading data from sqlite3")
conn = sqlite3.connect(args.sqlite_path)
snapshot = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name='board_climbs'").fetchone())
if snapshot:
    climbs = pd.read_sql_query("SELECT * FROM board_climbs WHERE board_type='kilter'", conn)
    stats = pd.read_sql_query("SELECT * FROM board_climb_stats WHERE board_type='kilter'", conn)
    metadata_conn = sqlite3.connect(args.metadata_path)
else:
    climbs = pd.read_sql_query("SELECT * FROM climbs", conn)
    stats = pd.read_sql_query("SELECT * FROM climb_stats", conn)
    metadata_conn = conn
grades = pd.read_sql_query("SELECT * FROM difficulty_grades", metadata_conn).set_index("difficulty")
holds = pd.read_sql_query("SELECT * FROM holes", metadata_conn)
placements = pd.read_sql_query("SELECT * FROM placements", metadata_conn)
holds = pd.merge(placements, holds, left_on="hole_id", right_on="id")
holds.set_index("id_x", inplace=True)

# merge and rename
df = pd.merge(climbs.drop("angle", axis=1), stats, left_on="uuid", right_on="climb_uuid")
if snapshot:
    legacy_climbs = pd.read_sql_query("SELECT uuid, frames, created_at FROM climbs WHERE layout_id=1", metadata_conn)
    legacy_stats = pd.read_sql_query(
        "SELECT c.uuid, c.frames, s.angle, s.ascensionist_count, s.quality_average, s.difficulty_average "
        "FROM climbs c JOIN climb_stats s ON c.uuid=s.climb_uuid WHERE c.layout_id=1 AND s.ascensionist_count>0",
        metadata_conn,
    )
    df = restore_legacy_stats(df, legacy_stats)
    print(f"Restored archived stats for {(df['stats_source'] == '2024').sum()} reset climb-angle rows")
df = df[df["difficulty_average"].notna()]
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
if args.broad:
    df = df[
        (df["ascensionist_count"] == 0)
        | ((df["ascensionist_count"] >= args.min_ascents) & (df["quality_average"] >= args.min_quality))
    ]
    print(f"Keeping unrated and positively climbed quality >= {args.min_quality}:\n{df.shape[0]}")
else:
    df = df[(df["quality_average"] >= args.min_quality) & (df["ascensionist_count"] >= args.min_ascents)]
    print(f"Keeping climbed quality >= {args.min_quality}:\n{df.shape[0]}")
df = df.reset_index()

holds = holds[holds["layout_id"] == 1]  # only original boards
holds = holds[holds.index.to_series() < 1800]


kp = KilterPolice(Tokenizer(), args.start_holds, args.finish_holds, args.foot_holds, args.hand_holds, args.total_holds)
df["valid"] = df["frames"].apply(kp.check)
df[~df["valid"]].to_csv(args.out_dir / "invalid_climbs.csv")
df = df[df["valid"]]
print(f"Removing invalid climbs, they are saved to 'invalid_climbs.csv':\n{df.shape[0]}")

df = deduplicate_climbs(df)
print(f"After deduplicating by holds/roles and angle:\n{len(df)}")

holds.to_csv(args.out_dir / "holds.csv")
grades.to_csv(args.out_dir / "grades.csv")

print("Splitting the data")
train, val, test = split_by_climb(df)
train.to_csv(args.out_dir / "train.csv")
val.to_csv(args.out_dir / "val.csv")
test.to_csv(args.out_dir / "test.csv")
if snapshot:
    test_post_2024 = post_archive_holdout(test, legacy_climbs)
    test_post_2024.to_csv(args.out_dir / "test-post-2024.csv")
    print(f"Post-archive, previously unseen test climbs: {len(test_post_2024)}")


### for plotter uses
# print("Creating image coordinates")
# holds['img_x'] = (7.5 * holds['x']).astype(int)
# holds['img_y'] = (-7.5 * holds['y'] + 1171).astype(int)
# holds[['img_x', 'img_y']].to_csv("figs/image_coords.csv")
# !wget https://raw.githubusercontent.com/Declan-Stockdale-Garbutt/KilterBoard_climb_generator/main/data/full_board_commercial.png
# !mv full_board_commercial.png figs/
# !convert figs/full_board_commercial.png -define png:color-type=2 figs/full_board_commercial.png
