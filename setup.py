import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Chicken-disease-Mlops"
AUTHOR_USER_NAME = "am-alibee"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "reachalibee@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="A Python package for CNN based chicken Disease Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)

