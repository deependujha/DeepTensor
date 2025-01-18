from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import toml


def get_pyproject_version(pyproject_path):
    try:
        with Path.open(pyproject_path) as file:
            pyproject_data = toml.load(file)
        return pyproject_data.get("project", {}).get("version")
    except Exception as e:
        print(f"Error reading {pyproject_path}: {e}")  # noqa: T201
        return None


def get_version_file_version(version_file_path):
    try:
        with Path.open(version_file_path, "r") as file:
            file_content = file.read()
        # Parse the file and extract version
        tree = ast.parse(file_content, filename=version_file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "version"
                        and isinstance(node.value, ast.Constant)
                    ):  # For Python 3.8+
                        return node.value.value
        print(f"Version not found in {version_file_path}")  # noqa: T201
        return None
    except Exception as e:
        print(f"Error reading {version_file_path}: {e}")  # noqa: T201
        return None


def main():
    pyproject_path = "pyproject.toml"
    version_file_path = os.path.join("src", "deeptensor", "__version__.py")  # noqa: PTH118

    pyproject_version = get_pyproject_version(pyproject_path)
    version_file_version = get_version_file_version(version_file_path)

    if pyproject_version is None or version_file_version is None:
        print("Error: Unable to fetch version(s).")  # noqa: T201
        sys.exit(1)

    if pyproject_version == version_file_version:
        print("Version check passed!")  # noqa: T201
        sys.exit(0)
    else:
        print(  # noqa: T201
            f"Version mismatch: pyproject.toml ({pyproject_version}) != __version__.py ({version_file_version})"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
