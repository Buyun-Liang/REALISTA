# REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations

This repository is the official implementation of the ICML 2026 paper [*REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations*](#).

Authors: [Buyun Liang](https://buyunliang.org/), [Jinqi Luo](https://peterljq.github.io/), [Liangzu Peng](https://liangzu.github.io/), [Kwan Ho Ryan Chan](https://ryanchankh.github.io/), [Darshan Thaker](https://darshanthaker.github.io/), [Kaleab A. Kinfu](https://kaleab.me/), [Fengrui Tian](https://tianfr.github.io/), [Hamed Hassani](https://www.seas.upenn.edu/~hassani/), and [René Vidal](https://www.grasp.upenn.edu/people/rene-vidal/).

[Project Website](https://github.com/Buyun-Liang/REALISTA) · [ArXiv](https://arxiv.org/abs/2605.12813) · [ICML Page](https://icml.cc/virtual/2026/poster/66287) · [Code](https://github.com/Buyun-Liang/REALISTA) · [Poster](realista_poster.png) <!-- · [License](./LICENSE) -->

[![Website](https://img.shields.io/badge/Website-REALISTA_Project-6f42c1)](https://github.com/Buyun-Liang/REALISTA)
[![arXiv](https://img.shields.io/badge/arXiv-2605.12813-b31b1b)](https://arxiv.org/abs/2605.12813)
[![ICML](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/virtual/2026/poster/66287)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Buyun-Liang/REALISTA)
[![Poster](https://img.shields.io/badge/Poster-PNG-ff69b4)](realista_poster.png)
<!-- [![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE) -->

## ✨ Abstract

⚠️ **Warning:** This method could be misused for malicious purposes.

Large language models (LLMs) achieve strong performance across many tasks but remain vulnerable to hallucinations, motivating the need for realistic adversarial prompts that elicit such failures. We formulate hallucination elicitation as a constrained optimization problem, where the goal is to find semantically coherent adversarial prompts that are equivalent to benign user prompts. Existing methods remain limited: discrete prompt-based attacks preserve semantic equivalence and coherence but search only over a limited set of prompt variations, while continuous latent-space attacks explore a richer space but often decode into prompts that are no longer valid rephrasings. To address these limitations, we propose **REALISTA**, a realistic latent-space attack framework. REALISTA constructs an input-dependent dictionary of valid editing directions, each corresponding to a semantically equivalent and coherent rephrasing, and optimizes continuous combinations of these directions in latent space. This design combines the optimization flexibility of continuous attacks with the semantic realism of discrete rephrasing-based attacks. Experiments demonstrate that REALISTA achieves superior or comparable performance to state-of-the-art realistic attacks on open-source LLMs and, crucially, succeeds in attacking large reasoning models under free-form response settings, where prior realistic attacks fail.

## 🖼️ Overview

![REALISTA Attack Generation](teaser_figure.png)

**Figure 1. Illustrative example of attack generation in REALISTA.** Starting from the original prompt $x_0$, the encoder $\phi$ maps it to its latent representation $z_0$. A perturbation composed from edit directions is added to obtain $z$, which is then decoded by $\psi$ back into the prompt space. The resulting adversarial prompt $x$ remains both semantically coherent and semantically equivalent to the original $x_0$, while inducing a hallucination.

![REALISTA Framework](framework_figure_new.png)

**Figure 2. Framework Overview.** (Left) *Input-dependent edit dictionary construction*. We employ a concept optimization procedure to construct a set of latent concepts ${c^{(1)}, \ldots, c^{(n)}}$ conditioned on the original prompt $x_0$ and WordNet. These concepts are assembled into an edit dictionary $D$, where each column corresponds to an interpretable editing direction $z^{(i)} = c^{(i)} - z_0$. (Right) *REALISTA overview*. REALISTA optimizes the editing strength vector $\delta$ and projects it onto a scaled latent simplex at each iteration. This latent simplex constraint is critical for preserving semantic equivalence between the original prompt and the adversarial prompt. The optimized $\delta$ is then used to construct the adversarial latent representation $z$.



## 🚧 Code Status

The code for this repository is currently **under construction**. We are preparing a clean and reproducible implementation of REALISTA, including:

- input-dependent edit dictionary construction;
- latent encoding and prompt reconstruction;
- REALISTA optimization over the scaled latent simplex;
- hallucination evaluation protocols.

Please check back later for updates.

## 📬 Contact

For questions or bug reports, please either:

- open an issue in this GitHub repository, or
- email [Buyun Liang](https://buyunliang.org/) at `byliang [at] seas [dot] upenn [dot] edu`.

## 📄 License

The code will be released under the MIT License. See [LICENSE](./LICENSE) for details.

## 🛡️ Disclaimer

The code and documentation in this repository are made available for research and educational purposes only, with no warranties or guarantees. Users are fully responsible for ensuring their work complies with applicable laws, regulations, and ethical standards. The authors disclaim any liability for misuse, damage, or harm resulting from the use of this material.
