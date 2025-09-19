"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Define the root source directory
src = Path(__file__).parent.parent / "src"

# Process all Python files in the source directory
for path in sorted(src.rglob("*.py")):
    # Skip test files and __pycache__
    if "test" in str(path) or "__pycache__" in str(path):
        continue
        
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __main__ files
    if parts[-1] == "__main__":
        continue
        
    # Handle __init__ files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    
    # Skip empty parts (shouldn't happen but just in case)
    if not parts:
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Create the documentation file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    # Set the edit path
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the navigation
with mkdocs_gen_files.open("api/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
