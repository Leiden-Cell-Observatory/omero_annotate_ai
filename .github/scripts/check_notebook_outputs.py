"""Fail if any committed Jupyter notebook contains outputs or execution counts.

Outputs can carry server names, user names and local paths, and they make diffs
unreadable. The nbstripout pre-commit hook strips them locally (`pixi run
install-hooks`); this check catches commits made without the hook.
"""

import json
import subprocess
import sys


def main() -> int:
    paths = subprocess.run(
        ["git", "ls-files", "*.ipynb"], capture_output=True, text=True, check=True
    ).stdout.split()
    dirty = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            notebook = json.load(f)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code" and (
                cell.get("outputs") or cell.get("execution_count") is not None
            ):
                dirty.append(path)
                break
    for path in dirty:
        print(f"{path}: has outputs; strip them (nbstripout {path}) and commit again")
    return 1 if dirty else 0


if __name__ == "__main__":
    sys.exit(main())
