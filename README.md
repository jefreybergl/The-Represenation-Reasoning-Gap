# The Representation-Reasoning Gap

**Language Models Encode Spatial Relations They Fail to Reason Over**

This project investigates whether small language models (GPT-2 Medium, Pythia 1.4B, Pythia 2.8B) encode spatial position information as linear representations in their residual stream even when they fail at multi-step spatial reasoning tasks. We use synthesized 5x5 grid-world navigation scenarios, linear probes (ridge classifiers) on residual stream activations, and behavioral evaluation to quantify the dissociation between representation formation and reasoning ability.

## Setup

```bash
pip install -r requirements.txt
```

## Running

Open and run `experiment.ipynb` from top to bottom. The notebook generates all experimental data, runs all analyses, and saves publication-ready figures to `paper/figures/`.

**Compute requirements:** A single GPU with >= 8GB VRAM is sufficient. Pythia 2.8B is the largest model (~6GB). The notebook processes models sequentially and manages GPU memory between runs. The notebook is designed for Google Colab (Pro recommended).