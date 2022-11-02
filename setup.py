from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    readme_text = f.read()


setup(
    name="scip_routing",
    version="0.1.0",
    description="A basic VRPTW Branch-and-Price solver",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    author="Mohammed Ghannam",
    author_email="mohammad.m.ghannam@gmail.com",
    url="https://github.com/mmghannam/scip-routing",
    license="MIT License",
    packages=find_packages(exclude=("tests", "docs", "data", "notebooks", "examples")),
    install_requires=[
        "pyscipopt",
        "networkx",
    ],
)