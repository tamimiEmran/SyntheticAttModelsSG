# setup.py
from setuptools import setup, find_packages
import os

# Function to read requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found. Proceeding without installing dependencies from file.")
        return []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Remove comments and empty lines, strip whitespace
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements

# Read README for long description
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    print("Warning: README.md not found. Long description will be empty.")

setup(
    name='energy-theft-detection', # Package name
    version='0.1.0', # Initial version
    author='Your Name / Team Name', # Replace with your name/team
    author_email='your.email@example.com', # Replace with your email
    description='A framework for simulating and detecting energy theft using synthetic attacks and machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/energy-theft-detection', # Replace with your GitHub repo URL
    license='MIT', # Match your LICENSE file
    # Find packages within the 'src' directory
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    # Specify Python versions required
    python_requires='>=3.8',
    # Install dependencies from requirements.txt
    install_requires=parse_requirements('requirements.txt'),
    # Define command-line scripts
    entry_points={
        'console_scripts': [
            # Core Experiments (Refactored)
            'run-exp1=experiments.experiment1:main',
            'run-exp2-removal=experiments.experiment2:main',
            'run-exp3-comparison=experiments.experiment3:main',
            'run-exp4-transfer=experiments.experiment4_transferability:main',
            'run-exp5-robustness=experiments.experiment5_robustness:main',
            'run-exp6-trainsize=experiments.experiment6_training_size:main',
            'run-exp7-monthly=experiments.experiment7_monthly_eval:main',
            'run-exp2-additive=experiments.experiment2_additive:main', # Added additive experiment

            # Tuning Script
            'run-tuning=experiments.tuning:main', # Assuming tuning.py has a main() function

            # Utility Scripts (from scripts/ directory)
            'generate-attacked-ausgrid=scripts.save_attacked_ausgrid:main',
            'generate-attacked-sgcc=scripts.save_attacked_sgcc:main',
        ],
    },
    # Add classifiers for PyPI metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
    ],
    # include_package_data=False, # Explicitly set to False or omit
)
