import csv
import shutil
from datetime import datetime
from pathlib import Path


PATH = Path("data/features_embeddings.csv")


def main():
    if not PATH.exists():
        print(f"Missing: {PATH}")
        return

    backup = PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    tmp = PATH.with_suffix(".repaired.tmp")
    shutil.copy2(PATH, backup)

    with open(PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        width = len(header)

        rows_by_id = {}
        malformed = 0
        total = 0
        for row in reader:
            total += 1
            if not row or not row[0]:
                malformed += 1
                continue
            if len(row) != width:
                malformed += 1
                row = row[:width] + [""] * max(0, width - len(row))
            rows_by_id[str(row[0]).strip()] = row

    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_by_id.values())

    tmp.replace(PATH)
    print(f"Backup:    {backup}")
    print(f"Rows read:  {total}")
    print(f"Rows kept:  {len(rows_by_id)}")
    print(f"Repaired:   {malformed}")
    print(f"Saved:     {PATH}")


if __name__ == "__main__":
    main()
