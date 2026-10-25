"""Copy results.txt and doe_results.txt into README.md between the markers.

    python run_all.py | tee results.txt
    python doe.py | tee doe_results.txt
    python tools/readme_results.py
"""
from pathlib import Path

readme = Path("README.md").read_text()
for name, start, end in (("results.txt", "<!-- results:start -->", "<!-- results:end -->"),
                         ("doe_results.txt", "<!-- doe:start -->", "<!-- doe:end -->")):
    if not Path(name).exists():
        print(f"skipped {name} (not found)")
        continue
    body = Path(name).read_text().strip()
    block = "\n".join([start, "```text", body, "```", end])
    a, b = readme.index(start), readme.index(end) + len(end)
    readme = readme[:a] + block + readme[b:]
Path("README.md").write_text(readme)
print("README.md updated")
