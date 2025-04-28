# Energy Theft Detection

This repository contains the implementation of various synthetic attack models for energy theft detection and evaluation of their effectiveness on real-world energy theft datasets.

## Overview

Advanced Metering Infrastructure (AMI) plays a significant role in smart grid systems, but faces security challenges including vulnerabilities to energy theft. This project evaluates the effectiveness of synthetic attack models in detecting real-world energy theft, providing insights into their limitations and correlations.

The research examines two key questions:
1. How effective are synthetic attack models at detecting real-world energy theft?
2. What are the correlations between different synthetic attack models?

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-theft-detection.git
cd energy-theft-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Running Experiments

The experiments are organized according to the research questions outlined in the paper:

```bash
# Run experiments by the referenced research question in the article as follows
python -m experiments.RQ1
```

### Using Attack Models

```python
from src.attack_models.implementations import AttackModel1, AttackModel2
from src.data.loader import load_ausgrid_data

# Load data
data = load_ausgrid_data()

# Apply attack model
attacked_data = AttackModel1().apply(data, alpha=0.5)
```

## Datasets

The project uses two main datasets:

1. **SGCC Dataset**: The State Grid Corporation of China dataset, which contains daily electricity consumption of 42,372 consumers with 3,615 labeled as thieves.
   - Source: [SGCC Dataset](https://github.com/henryRDlab/ElectricityTheftDetection)

2. **Ausgrid Dataset**: Contains consumption data from 300 consumers over a three-year span with readings taken every 30 minutes.
   - Source: [Ausgrid Dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data)

## Attack Models

The project implements 14 different attack models that manipulate energy consumption data:

| Attack ID | Description |
|-----------|-------------|
| 0 | Reports zero consumption |
| 1 | Decreases consumption by constant factor |
| 2 | Zero consumption for random duration |
| 3 | Multiplies consumption by different random factors |
| 4 | Decreases consumption in random time period |
| 5 | Substitutes consumption with random proportion of the mean |
| 6 | Replaces values above cut-off point with the cut-off point |
| 7 | Replaces values below a cut-off point with zeros |
| 8 | Reduces usage progressively to max intensity |
| 9 | Replaces samples with average daily usage |
| 10 | Reverses consumption trend |
| 11 | Lowers consumption for certain time |
| 12 | Swaps consumption with lower-consuming user |
| 13 | Intermittent energy theft or malfunction |

For detailed descriptions of each attack model, see the [documentation](docs/attack-models-doc.md).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{altamimi2023effectiveness,
  title={How Effective are Synthetic Attack Models to Detect Real-World Energy Theft?},
  author={Altamimi, Emran and Al-Ali, Abdulaziz and Al-Ali, Abdulla K. and Aly, Hussein and Malluhi, Qutaibah M.},
  journal={Energy},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
