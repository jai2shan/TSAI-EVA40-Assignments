import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asgnmnt6", # Replace with your own username
    version="0.0.32",
    author="jai2shan",
    author_email="muralis2raj@gmail.com",
    description="HR Attrition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="Unknown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

##setuptools.