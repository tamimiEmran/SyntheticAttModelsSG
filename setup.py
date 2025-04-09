from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="energy-theft-detection",
    version="0.1.0",
    author="Emran Altamimi, Abdulaziz Al-Ali, Abdulla K. Al-Ali, Hussein Aly, Qutaibah M. Malluhi",
    author_email="ea1510662@qu.edu.qa",
    description="Energy theft detection using synthetic attack models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/energy-theft-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-experiment1=experiments.experiment1:main",
            "run-experiment2=experiments.experiment2:main",
            "run-experiment3=experiments.experiment3:main",
        ],
    },
)
