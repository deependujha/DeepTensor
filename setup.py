#!/usr/bin/env python  # noqa: EXE001
from __future__ import annotations

import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)  # noqa: PTH120
_PATH_SOURCE = os.path.join(_PATH_ROOT, "src")  # noqa: PTH118
_PATH_REQUIRES = os.path.join(_PATH_ROOT, "requirements")  # noqa: PTH118


def _load_py_module(fname, pkg="deeptensor"):
    spec = spec_from_file_location(
        os.path.join(pkg, fname),  # noqa: PTH118
        os.path.join(_PATH_SOURCE, pkg, fname),  # noqa: PTH118
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _load_requirements(
    path_dir: str = _PATH_ROOT, file_name: str = "requirements.txt"
) -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())  # noqa: PTH118, PTH123, SIM115
    return list(map(str, reqs))


about = _load_py_module("__about__.py")
with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fopen:  # noqa: PTH118, PTH123
    readme = fopen.read()


def _prepare_extras(
    requirements_dir: str = _PATH_REQUIRES, skip_files: tuple = ("docs.txt", "test.txt")
) -> dict:
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install lit-data[dev, docs]`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    req_files = [Path(p) for p in glob.glob(os.path.join(requirements_dir, "*.txt"))]  # noqa: PTH118, PTH207
    extras = {
        p.stem: _load_requirements(file_name=p.name, path_dir=str(p.parent))
        for p in req_files
        if p.name not in skip_files
    }
    # todo: eventually add some custom aggregations such as `develop`
    extras = {name: sorted(set(reqs)) for name, reqs in extras.items()}
    return extras  # noqa: RET504


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="deeptensor",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/deependujha/deeptensor",
    license=about.__license__,
    long_description=readme,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI", "streaming", "cloud", "model training"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=_load_requirements(),
    extras_require=_prepare_extras(),
    project_urls={
        "Bug Tracker": "https://github.com/deependujha/deeptensor/issues",
        "Documentation": "https://deependujha.github.io/deeptensor",
        "Source Code": "https://github.com/deependujha/deeptensor",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
