"""
Setup comments:
Install using:                pip install -e .
"""
from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path: str = "requirements.txt") -> list[str]:
    """
    Read a pip-style requirements file and return a list of requirement strings.
        - Ignores blank lines and comments.
        - Ignores editable (-e) and local path requirements (common in dev workflows).
    If you want to allow those, remove the relevant checks below.
    """
    req_path = Path(__file__).resolve().parent / path
    if not req_path.exists():
        raise FileNotFoundError(f"Could not find {req_path}")

    requirements: list[str] = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Optional: skip pip options that setuptools doesn't handle well in install_requires
        if line.startswith(("-r ", "--requirement ", "--find-links ", "-f ", "--index-url ", "--extra-index-url ")):
            continue

        # Optional: skip editable installs / local paths in install_requires
        if line.startswith(("-e ", "--editable ")):
            continue
        if line.startswith((".", "/", "file:")):
            continue

        requirements.append(line)
    return requirements


HERE = Path(__file__).resolve().parent
README = (HERE / "README.md").read_text(encoding="utf-8")
INSTALL_REQUIRES = read_requirements("requirements.txt")

setup(
    name="RP-SLAM",
    version="0.1.1",
    description="Recognition-parametrised simultaneous localisation and mapping.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="James Green",
    url="https://github.com/jamesgreen-c/RP-SLAM",
    license="MIT",
    packages=find_packages(),
    python_requires="==3.12.3",
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "jax",
        "recognition",
        "variational-optimisation",
        "expectation-maximisation"
        "slam",
    ],
)

