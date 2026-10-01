import sqlite3

DB = "data/tuneai.db"

db = sqlite3.connect(DB)

print("=" * 60)
print("TUNEAI SONGS TABLE STRUCTURE")
print("=" * 60)

columns = db.execute("PRAGMA table_info(songs)").fetchall()

for column in columns:
    cid, name, data_type, not_null, default_value, primary_key = column

    print(
        f"{cid:2} | "
        f"{name:25} | "
        f"{data_type:12} | "
        f"NOT NULL={not_null} | "
        f"PK={primary_key} | "
        f"DEFAULT={default_value}"
    )

print("=" * 60)

print("\nSample existing song:")

sample = db.execute(
    "SELECT * FROM songs LIMIT 1"
).fetchone()

for column, value in zip(columns, sample):
    print(f"{column[1]:25}: {value}")

db.close()