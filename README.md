<p align="center"><img src="https://github.com/pkargupta/cognitive_foundations/blob/main/figs/readme_image.png" alt="Cognitive Foundations"/></p>

![Static Badge](https://img.shields.io/badge/Paper-white?style=for-the-badge&logo=arxiv&logoColor=%23e46e2f&color=%232e4969&link=https%3A%2F%2Farxiv.org%2Fabs%2F2511.16660)
![Static Badge](https://img.shields.io/badge/Blog-white?style=for-the-badge&logo=notion&logoColor=%23e46e2f&color=%232e4969&link=https%3A%2F%2Ftinyurl.com%2Fcognitive-foundations)
![Static Badge](https://img.shields.io/badge/Dataset-white?style=for-the-badge&logo=huggingface&logoColor=%23e46e2f&color=%232e4969&link=https%3A%2F%2Fhuggingface.co%2Fcollections%2Fstellalisy%2Fcognitive-foundations)

# Cognitive Foundations for Reasoning and Their Manifestation in LLMs


## Overview

Our framework bridges **cognitive science** and **large language model (LLM) research** to systematically understand how LLMs reason and to diagnose/improve their reasoning processes, based on analysis of 192K model traces and 54 human think-aloud traces.

## Key Contributions

**Comprehensive Framework**: We develop a taxonomy of **28 cognitive elements** spanning reasoning goals & properties, meta-cognitive controls, reasoning & knowledge representations, and transformation operations, creating a shared vocabulary between cognitive science and LLM research. We utilize this framework to encode reasoning traces into a **heterogenous graph**, where each node represents a cognitive element and edges between them reflect their temporal and hierarchical relationships.

**Large-Scale Empirical Analysis**: Our evaluation encompasses **192,000 model traces** from **18 different LLMs** across text, vision, and audio modalities, alongside **54 human think-aloud traces** to enable direct comparison between human and machine reasoning patterns. We study both _well-structured_ (e.g., Algorithmic) to _ill-structured_ (e.g., Dilemma) problem types.

**Test-Time Reasoning Guidance**: We introduce **test-time reasoning guidance** as a targeted intervention to explicitly scaffold cognitive patterns predictive of reasoning success. In greedy fashion, we determine the most success-prone reasoning structure (subgraph) for each problem type, based on our empirical analysis. We convert each into a prompt which guides a model's reasoning process, improving performance by up to 26.7% on ill-structured problems while maintaining baseline performance on well-structured ones.

## Citation

```bibtex
@article{kargupta2025cognitive,
  title={Cognitive Foundations for Reasoning and Their Manifestation in LLMs},
  author={Kargupta, Priyanka and Li, Shuyue Stella and Wang, Haocheng and Lee, Jinu and Chen, Shan and Ahia, Orevaoghene and Light, Dean and Griffiths, Thomas L and Kleiman-Weiner, Max and Han, Jiawei and Celikyilmaz, Asli and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2511.16660},
  year={2025}
}
```