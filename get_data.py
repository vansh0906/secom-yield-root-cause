"""Download the UCI SECOM semiconductor manufacturing dataset into data/.

    python get_data.py

SECOM: 1,567 wafers from a real production line, 590 anonymised sensor/process measurements each,
a pass/fail result from in-line testing and a timestamp. Source: UCI Machine Learning Repository
(McCann & Johnston, 2008), CC BY 4.0.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).parent / "data"
FILES = ("secom.data", "secom_labels.data")
OLD = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
ZIP = "https://archive.ics.uci.edu/static/public/179/secom.zip"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    OUT.mkdir(exist_ok=True)
    if all((OUT / f).exists() for f in FILES):
        print("data/ already has both files")
        return 0
    try:
        for f in FILES:
            (OUT / f).write_bytes(fetch(OLD + f))
        print("downloaded from", OLD)
    except Exception as e1:
        try:
            z = zipfile.ZipFile(io.BytesIO(fetch(ZIP)))
            for f in FILES:
                (OUT / f).write_bytes(z.read(f))
            print("downloaded from", ZIP)
        except Exception as e2:
            print("Download failed:", e1, "|", e2)
            print("Download secom.zip by hand from https://archive.ics.uci.edu/dataset/179/secom")
            print("and put secom.data and secom_labels.data in the data/ folder.")
            return 1
    n = sum(1 for _ in open(OUT / "secom.data"))
    print(f"secom.data: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
