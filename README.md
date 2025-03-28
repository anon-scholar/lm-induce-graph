# Anonymous codebase for the COLM submission "Can an LM Induce a Graph? Investigating Memory Drift and Context Length"

## Organization of the repository
Our evaluation benchmark has three subtasks, edge, subgraph and clique discovery. Each has separate script and can be run separately. The code can be run with any LLM, local or through API access.

## Install a new conda environment using the following:
Note that we are using python=3.10.0 for now.
```console
conda create --prefix env python=3.10.0
conda activate env/
pip install -r requirements.txt
```

