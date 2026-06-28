"""
Run this from your project root:  python find_duplicates.py

It searches your whole project folder for every file named
streamlit_app.py, and for each one found, reports:
  - its full path
  - whether it contains the broken "_through" reference
  - its file size and last-modified time

This will show us if there's a second copy being picked up instead
of the one you uploaded.
"""
import os
import time

ROOT = os.getcwd()
found = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    # Skip virtual env / git internals — not relevant and very slow to walk
    dirnames[:] = [d for d in dirnames if d not in (".venv", "venv", ".git", "__pycache__", "node_modules")]
    for fname in filenames:
        if fname == "streamlit_app.py":
            full = os.path.join(dirpath, fname)
            found.append(full)

if not found:
    print("No streamlit_app.py found anywhere under", ROOT)
else:
    print(f"Found {len(found)} copy(ies) of streamlit_app.py:\n")
    for path in found:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            has_bug = "_through" in content
            size = os.path.getsize(path)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
            print(f"  Path:     {path}")
            print(f"  Size:     {size} bytes")
            print(f"  Modified: {mtime}")
            print(f"  Contains '_through' bug: {has_bug}")
            print()
        except Exception as e:
            print(f"  Path: {path}  -> ERROR reading file: {e}\n")
