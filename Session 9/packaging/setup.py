import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asgnmt9", # Replace with your own username
    version="0.0.6",
    author="TSAI-Assignment9",
    author_email="muralis2raj@gmail.com",
    description="Assignment9",
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