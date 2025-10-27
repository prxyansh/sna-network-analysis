# SNA Project

This project loads the Email-EU-core network and renders:
- A small subgraph visualization
- A degree distribution histogram

## Prerequisites

- Python 3.10+ (project venv already configured in `.venv` if you opened this folder in VS Code)
- Dependencies: `networkx`, `matplotlib`, `pandas` (see `requirements.txt`)

## Setup

Using the workspace virtual environment (recommended):

```zsh
cd /Users/prx./Documents/SNA/Project_finl
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you prefer not to activate the venv:

```zsh
/Users/prx./Documents/SNA/Project_finl/.venv/bin/python -m pip install -r requirements.txt
```

## Run

Save plots to files (default):

```zsh
python project.py
```

Specify output paths and sample size:

```zsh
python project.py --out subgraph.png --out-degree degree_distribution.png --sample-n 100
```

Show interactively instead of saving (requires GUI):

```zsh
python project.py --show
```

The dataset file `email-Eu-core.txt` is expected to be in the same folder as `project.py`. The script resolves the path relative to its own location, so you can run it from any working directory.

Outputs by default:
- `subgraph.png` — subgraph visualization
- `degree_distribution.png` — histogram of node degrees
