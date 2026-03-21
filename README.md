# The Representation-Reasoning Gap

**Language Models Encode Spatial Relations They Fail to Reason Over**

Do small language models actually represent spatial information internally, even when they can't answer spatial questions correctly? That's what this project tries to figure out. We probe GPT-2 Medium, Pythia 1.4B, and Pythia 2.8B on synthesized 5x5 grid-world navigation scenarios. We fit ridge classifiers (linear probes) to residual stream activations and compare what the models "know" internally against their behavioral performance. Turns out there's a pretty clear gap between what gets encoded and what the model can actually use for multi-step reasoning.

## Setup

```bash
pip install -r requirements.txt
```

## Running

Just open `experiment.ipynb` and run it top to bottom. It generates the data, runs all analyses, and saves figures to `paper/figures/`.

**Compute requirements:** Any GPU with 8GB+ VRAM will work. Pythia 2.8B is the biggest model at ~6GB. Models are loaded one at a time. Built to run on Google Colab (Pro recommended).
