from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

def setup_package():
    setup(
        name="pybees", 
        version="0.0.3",
        author="Joseph Moorhouse",
        author_email="moorhouse@live.co.uk",
        description="A research toolkit for the Bees algorithm in Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jbmoorhouse/pybees",
        packages=find_packages(include=["pybees", "pybees.*"]),
        install_requires = [
            "numpy>=1.17.4",
            "scipy>=1.3.2",
            "plotly>=4.4.1",
            "tqdm>=4.40.2",
            "scikit-learn>=0.22",
            "pandas>=0.25.3"
        ],
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering"
        ],
        python_requires='>=3.6'
    )

if __name__ == "__main__":
    setup_package() 