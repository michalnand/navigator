import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LibsNavigator",
    version="1.0.0",
    author="Michal Chovanec",
    author_email="michal.nand@gmail.com",
    description="visual robot navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)