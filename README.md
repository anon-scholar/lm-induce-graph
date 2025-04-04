# Anonymous codebase for the COLM submission "Can an LM Induce a Graph? Investigating Memory Drift and Context Length"

## Organization of the repository
Our evaluation benchmark has three subtasks, edge, subgraph and clique discovery. Each has separate script and can be run separately.

## Install a new conda environment using the following:
Note that we are using python=3.10.0 for now.
```console
conda create --prefix env python=3.10.0
conda activate env/
pip install -r requirements.txt
```

## Implementation details
The evaluation framework is designed to use any type of LLM, both through API and local storage. We experimented with four models, two closed-source and two open-source to demonstrate the effect of memory drift in graph reconstruction task : i) GPT-4o from OpenAI, ii) Llama-3 from Meta AI, iv) Mistral-7B from MistralAI, and v) Gemini-2 from Gemini platform, Google. While the closed-source models are accessed via their APIs, other models were used on a compute cluster of 4 Nvidia Tesla P100s 16GB each. The input to the framework are two files, i) an adjacency matrix with 0/1 indicating connection. ii) The description of the nodes in json format. See `docs/prompt.md` for prompt templates.
