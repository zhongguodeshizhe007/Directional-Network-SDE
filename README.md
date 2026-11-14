# NetworkSDE: Symbolic Dynamics Discovery and Coherent Probabilistic Early Warning of Freeway Conflict Risk Propagation

Official code repository for the paper:
> **From Black-Box Networks to Explicit Governing Laws: Symbolic Dynamics Discovery and Coherent Probabilistic Early Warning of Freeway Conflict Risk Propagation**  
> *Transportation Research Part C: Emerging Technologies*

## Overview

This repository provides an end-to-end framework uniting stochastic continuous network physics with discrete conflict risk early warning on freeway corridors:
1. **Continuous Network SDE**: Formulates directional Itô network stochastic differential equations over multi-lane corridors with heteroskedastic diffusion and left-censored likelihoods.
2. **STLSQ Symbolic Discovery**: Distills five Lyapunov-stable explicit governing laws with bounded shock response.
3. **Zero-Inversion Survival Hazard Early Warning**: Structural multi-horizon survival calibration that completely eliminates cross-horizon probability inversion (0.00% violation).
4. **Sub-millisecond V2X Edge Inference**: Highly optimized runtime achieving 0.42 ms CPU execution within 4.5 MB memory for connected vehicle alerting.

## Directory Structure

```text
Github/
├── 00_SelfTest.ipynb         # Zero-data software self-test with synthetic data
├── 01_MainWorkflow.ipynb     # Main stage-by-stage pipeline workflow
├── 02_PaperWorkflow.ipynb    # Full paper experiments, training, and figure generation
├── configs/                  # Experiment configuration and hyperparameters
│   └── experiment.json
├── docs/                     # Methodological definitions, GPU guide, and research scope
├── highd_nb/                 # Core Python package (NetworkSDE)
│   ├── core/                 # SDE models, symbolic regression, preprocessing, evaluation
│   ├── paper/                # Survival hazard layer, baselines, SVGP, plotting reports
│   └── vendor/               # Synthetic data fixtures and dataset preparation tools
├── tests/                    # Automated test suite (161 unit & integration tests)
├── tools/                    # Command-line entrypoints (doctor.py, run_research.py, etc.)
├── data/                     # Dataset placeholder (instructions for highD data placement)
├── highd-dataset-v1.0/       # Standard highD layout placeholder
├── requirements.txt          # Python dependencies
├── setup_windows.ps1         # Automated Windows environment installer
└── start_jupyter.ps1         # Jupyter Lab launcher
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/NetworkSDE.git
cd NetworkSDE

# Install dependencies
pip install -r requirements.txt
# Install PyTorch matching your CUDA version (or CPU):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

On Windows PowerShell, you can simply run:
```powershell
.\setup_windows.ps1
```

### 2. Verify Environment (Zero-Data Doctor)

Run the self-contained diagnostic tool to check PyTorch, CUDA, and scientific kernels:
```bash
python tools/doctor.py --device auto
```

### 3. Run Self-Test

Open `00_SelfTest.ipynb` in Jupyter, or run automated unit tests:
```bash
pytest tests
```

### 4. Running Full Paper Experiments

1. Download the highD dataset from [highD-dataset.com](https://highd-dataset.com/) and place CSV recordings into `highd-dataset-v1.0/data/`.
2. Open `02_PaperWorkflow.ipynb` to execute the full pipeline and reproduce paper figures.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
