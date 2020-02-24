from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.in") as f:
    requirements = f.read().splitlines()

def setup_package():

    setup(
        name="pybees", 
        version="0.0.1",
        author="Joseph Moorhouse",
        author_email="moorhouse@live.co.uk",
        description="A research toolkit for the Bees algorithm in Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jbmoorhouse/pybees",
        packages=find_packages(include=["pybees", "pybees.*"]),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD 3 License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )

if __name__ == "__main__":
    setup_package() 