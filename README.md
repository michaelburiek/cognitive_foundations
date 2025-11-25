<p align="center"><img src="https://github.com/pkargupta/cognitive_foundations/blob/main/figs/readme_image.png" alt="Cognitive Foundations"/></p>

# Cognitive Foundations for Reasoning and Their Manifestation in LLMs


## Overview

Our framework bridges **cognitive science** and **large language model (LLM) research** to systematically understand how LLMs reason and to diagnose/improve their reasoning processes, based on analysis of 192K model traces and 54 human think-aloud traces.

## Key Contributions

**Comprehensive Framework**: We develop a taxonomy of **28 cognitive elements** spanning reasoning goals & properties, meta-cognitive controls, reasoning & knowledge representations, and transformation operations, creating a shared vocabulary between cognitive science and LLM research. We utilize this framework to encode reasoning traces into a **heterogenous graph**, where each node represents a cognitive element and edges between them reflect their temporal and hierarchical relationships.

**Large-Scale Empirical Analysis**: Our evaluation encompasses **192,000 model traces** from **18 different LLMs** across text, vision, and audio modalities, alongside **54 human think-aloud traces** to enable direct comparison between human and machine reasoning patterns. We study both _well-structured_ (e.g., Algorithmic) to _ill-structured_ (e.g., Dilemma) problem types.

**Test-Time Reasoning Guidance**: We introduce **test-time reasoning guidance** as a targeted intervention to explicitly scaffold cognitive patterns predictive of reasoning success. In greedy fashion, we determine the most success-prone reasoning structure (subgraph) for each problem type, based on our empirical analysis. We convert each into a prompt which guides a model's reasoning process, improving performance by up to 26.7% on ill-structured problems while maintaining baseline performance on well-structured ones.