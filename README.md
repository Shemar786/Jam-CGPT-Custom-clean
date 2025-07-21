# Jam-CGPT: Distilled GPT for Source Code Summarization

## Code for Distilled GPT for Source Code Summarization

Proposed by:
- [Chia-Yi Su](https://chiayisu.github.io/)
- [Collin McMillan](https://sdf.org/~cmc/)

This repository contains all the code and detailed instructions to rebuild [Jam-CGPT](https://huggingface.co/apcl/Jam-CGPT) models in our HuggingFace [Automatic Program Comprehension Lab](https://huggingface.co/apcl) hub.

## Jam‑CGPT — a Personal, Lightweight LLM

> Jam‑CGPT is a trimmed‑down GPT‑2 variant that runs happily on a single workstation. Point it at a new database schema, fire off a quick two‑hour retraining run, and you’re back in business.

### Why Jam‑CGPT?

* **Data never leaves your machine** — ideal for sensitive projects.  
* **Easy experimentation** with niche or proprietary datasets.  
* **Rapid schema swaps** without multi‑day training marathons.

### At a Glance

| Strengths | Trade‑offs |
|-----------|-----------|
| Near‑zero cost per query | Requires a short tuning pass |
| Runs fully offline | A few points shy of GPT‑4o accuracy |
| Minimal operating cost | No web UI (CLI workflow) |
| Retrainable for each project | |

### Model Specs

* **Parameters:** ≈ 50 M  
* **Hardware:** CPU‑only—no GPU required  
* **Retargeting time:** ≈ 2 h for a new schema

### Training Recipe

| Step | Dataset Size | Goal |
|------|--------------|------|
| Generic text pre‑training | ~1 B lines | Core language understanding |
| Stack Overflow fine‑tune | ~13 M Q&A pairs | Tech jargon & code patterns |
| English → SQL supervision | ~8 K pairs | Natural‑language query generation |

## Citation
This work was accepted to [Automated Software Engineering](https://link.springer.com/journal/10515), an academic journal.  If you use this work in an academic paper, please cite the following:
```
@misc{su2024distilled,
      title={Distilled GPT for Source Code Summarization}, 
      author={Chia-Yi Su and Collin McMillan},
      year={2024},
      journal={Automated Software Engineering}
}
```
Preprint PDF available here: https://arxiv.org/abs/2308.14731

