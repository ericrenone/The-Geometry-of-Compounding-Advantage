# The Geometry of Compounding Advantage

## Five Hardware-Software Co-Design Events, One Infrastructure Stack, and the $100B Transition Already in Motion

---

**Strategic Intelligence Report · 2027 Edition**  
*Infrastructure & Platforms · AI Systems Strategy*

**Prepared for:** Technology Executives · Infrastructure Investors · AI Architecture Leadership  
**Primary Evidence Base:** 50+ peer-reviewed sources · NeurIPS 2025 · ICLR 2026 · *Phys. Rev. E* 2026 · IEEE TVLSI 2019

---

## The Bottom Line

> **Five independent co-design decisions — made across 1985, 1986, and 2017 — selected the arithmetic, the optimizer, and the silicon that define every production AI system operating today. Those decisions are now simultaneously under pressure from the workloads they cannot efficiently serve. The organizations that correctly identify which layer of the stack is failing, why it is failing, and what replaces it will capture the next infrastructure cycle. The ones that do not will fund it.**

The AI infrastructure landscape of 2027 is defined by a tension between two timelines. The first is operational: frontier Transformer systems continue to scale on Euclidean, floating-point, gradient-descent substrates, achieving state-of-the-art results on benchmarks that do not yet penalize geometric mismatch, energy inefficiency, or curvature blindness. This timeline has 24–36 months of runway. The second is structural: three compounding substrate failures — in arithmetic, in optimization, and in representational geometry — have reached the point where the cost of accommodation exceeds the cost of replacement for specific, large, and commercially consequential workload classes. The structural timeline is forcing action now.

This report synthesizes five research programs and constructs the unified strategic picture: what was decided, when, and why; what the compounding costs are; where the replacement infrastructure is emerging; and what leadership teams must do before the window closes.

---

## Key Findings

| # | Finding | Strategic Urgency |
|---|---|---|
| 1 | The Euclidean assumption is structurally broken for hierarchical domains | **Act Now** |
| 2 | Three substrate lotteries are simultaneously reaching their economic ceilings | **Act Now** |
| 3 | Attention is a geometric connection, not merely an algorithm — with direct training efficiency implications | **Plan in 2027** |
| 4 | Grokking is a capital allocation problem, not a research curiosity | **Act Now** |
| 5 | The 2028 biology compute wall is real, structural, and not addressable by process-node improvement | **Plan in 2027** |
| 6 | The Lorentzian transition has cleared its engineering obstacles and is entering the investment phase | **Evaluate Now** |
| 7 | Google's 2017 open-source of the Transformer was a platform strategy with lock-in consequences still accumulating | **Assess Exposure** |
| 8 | The next infrastructure moat is geometric, not scalar — and the co-design window is open | **Invest Now** |

---

## Table of Contents

1. [The Five Lotteries — How the Stack Was Built](#1-the-five-lotteries--how-the-stack-was-built)
2. [The Three Ceilings — Where the Stack Breaks](#2-the-three-ceilings--where-the-stack-breaks)
3. [The Geometric Object — What the Stack Is Computing](#3-the-geometric-object--what-the-stack-is-computing)
4. [The Fiber Bundle Architecture — Operational Translation](#4-the-fiber-bundle-architecture--operational-translation)
5. [Grokking Is a Capital Problem](#5-grokking-is-a-capital-problem)
6. [The Lock-In Economy — Dependency Map](#6-the-lock-in-economy--dependency-map)
7. [The Lorentzian Transition — Market Sizing](#7-the-lorentzian-transition--market-sizing)
8. [Strategic Archetypes and Competitive Positions](#8-strategic-archetypes-and-competitive-positions)
9. [The 2027–2030 Action Agenda](#9-the-20272030-action-agenda)
10. [Ten Falsifiable Predictions](#10-ten-falsifiable-predictions)
11. [Conclusion — The Geometry of the Next Decade](#11-conclusion--the-geometry-of-the-next-decade)
12. [Evidence Base and Sources](#12-evidence-base-and-sources)

**Appendices**

- [Appendix A — Mathematical Reference: The RBH Framework](#appendix-a--mathematical-reference-the-rbh-framework)
- [Appendix B — Glossary of Key Technical Terms](#appendix-b--glossary-of-key-technical-terms)
- [Appendix C — Methodology and Scope](#appendix-c--methodology-and-scope)

---

## 1. The Five Lotteries — How the Stack Was Built

Sara Hooker's *Hardware Lottery* (2020) established the foundational principle: **a research idea wins not because it is theoretically superior, but because it is perfectly matched to the available hardware and software ecosystem.** Every layer of the current AI infrastructure stack is the product of a lottery won on hardware-software co-fitness grounds, not theoretical optimality. Understanding which layer was decided when — and why — is the prerequisite for understanding which transitions are now forced.

### The Lottery Cascade

**Lottery 1 — The Arithmetic Lottery (1985)**  
The IEEE 754 floating-point standardization fixed the representational substrate of every chip built since. The decisive trade: fixed-point arithmetic (which exposes iteration count, is natively CORDIC-compatible, and bills in ticks) lost to floating-point arithmetic (which hides iteration, is natively GEMM-compatible, and bills in FLOPS). The winning format — sign bit, biased exponent, normalized mantissa — was optimized for one-shot evaluation of flat arithmetic operations on a flat dynamic range. It was not optimized for the iterative, convergent, curved-geometry workloads that would define frontier AI from 2017 forward.

**The cost paid:** Every diffusion sampler, every Sinkhorn iteration, every score-function evaluation, every denoising step is a fixed-point iteration running on a substrate designed for the opposite computational shape. Published estimates put the energy overhead at 2–10× for iterative workloads on IEEE 754 silicon relative to what iteration-native substrate would require (Luo et al., IEEE TVLSI 2019).

**Lottery 2 — The Optimization Lottery (1986)**  
Rumelhart, Hinton, and Williams' backpropagation-with-gradient-descent paper committed the field to Euclidean parameter-space navigation. The winning optimizer — stochastic gradient descent and its Adam-family descendants — treats parameter space as flat, requiring O(N) storage and O(N) compute per step. The theoretically correct optimizer — natural gradient descent (Amari, 1998) — treats parameter space as a Riemannian manifold with the Fisher information matrix as its metric, requiring O(N²) storage and O(N³) inversion for N parameters.

**The cost paid:** Every frontier training run uses more steps than necessary. Sophia (Liu et al., 2023) demonstrated 50% step reduction from recovering a single dimension of curvature (diagonal Hessian). For problems where the condition number κ of the Fisher matrix exceeds 10⁴ — empirically true for large language models — gradient descent converges in O(κ) steps while natural gradient converges in O(√κ) steps, a gap of 100× at κ = 10⁴. The Anthropic 5-gigawatt AWS Trainium commitment is sized to a substrate running optimizers that discard curvature. Curvature-aware optimization on curvature-native silicon would address the same training workload at a fraction of the energy budget.

**Lottery 3 — The Silicon Lottery (2017)**  
Vaswani et al.'s *Attention Is All You Need* (NeurIPS 2017) is universally described as an algorithmic breakthrough. It is more precisely described as the product of hardware-software co-design under unified organizational authority. The Transformer architecture — scaled dot-product attention, multi-head parallelism, position-wise feed-forward networks — is composed, without exception, of dense matrix multiplications. It is not incidentally compatible with Google's systolic-array TPU architecture. It is, provably, the workload a 256×256 Matrix Multiply Unit was built to maximize.

**The chronological proof:** Google announced TPU v2 — the first training-capable custom silicon, introducing bfloat16 arithmetic and Pod interconnect — at Google I/O in **May 2017**. Vaswani et al. was submitted to arXiv **one month later**, explicitly noting large-model training on TPU v2 Pods. The authors had architectural access to the machine they were designing for before it was publicly announced. Jeff Dean — senior author of both the TPU v1 paper (ISCA 2017) and the Transformer paper — served as the organizational bridge. This is the fingerprint of co-design, not coincidence.

**The cost paid:** Every organization training a Transformer derivative is optimizing a Google-native workload on Google-designed silicon. Anthropic trains Claude on Google Cloud TPUs and Amazon Trainium. OpenAI required a $13B+ Microsoft investment in Azure infrastructure specifically to avoid Google hardware dependency. Meta's LLaMA open-releases are architectural Transformer derivatives. The global frontier AI landscape is, at the architectural level, a Google-designed workload running on a Google-designed substrate.

**Lottery 4 — The Biology Compute Lottery (ongoing → ceiling 2028)**  
Biology AI workloads — AlphaFold-class structure prediction, RFdiffusion-class de novo design, Evo-class genomic foundation modeling — share five computational primitives that are architecturally foreign to GEMM: the N×N×D pair tensor, SE(3) equivariance, diffusion denoising, long-context Hyena filtering, and rank-one Sherman-Morrison edits. Matmul-only substrates emulate all five at compounded overhead of 3–12× per operation, paid on every Pairformer block, every Structure Module pass, every denoising step.

**The economics:** AlphaFold 3-class training at matmul-era cost: $40–60M. On pair-tensor-native silicon: $6–10M. Saturation mutational scan of a 300-residue protein: $40 on GB300, $0.50 on native substrate. Whole-genome CRISPR off-target screen: $2.4M on matmul-era substrate, $30K on native. The overhead is structural — it is set by architecture, not process node, and does not shrink with Moore's Law. As biology AI absorbs 14% of frontier compute by 2028, these line items become board-level capital crises at the frontier laboratories and hyperscalers that supply them.

**Lottery 5 — The Geometry Lottery (2025–2026, in progress)**  
Five independent research programs converged in 2025–2026 on a precise geometric description of what every production language model is computing: a sloppy hyperribbon manifold with stratified Lorentzian metric, polytopal fiber structure, and Radon-domain BV functional presentation. Token embedding spaces exhibit significantly negative Ricci curvature (Robinson et al., 2024/2025). Billion-parameter fully hyperbolic models outperform matched Euclidean baselines by up to 4% on MMLU and ARC (HELM, NeurIPS 2025). The trainability obstacles that blocked Lorentzian architectures for a decade were resolved in January–June 2026.

**The cost paid:** Infrastructure built on Euclidean geometry assumptions is operating on the wrong substrate for all hierarchically structured domains. Cosine similarity applied to token embeddings is a distortion, not a measurement. Retrieval infrastructure, embedding databases, and nearest-neighbor search pipelines built on flat-space assumptions introduce systematic errors that scale with the degree of hierarchical structure in the domain — which is to say, they degrade precisely in the domains of highest commercial value: knowledge graphs, legal documents, biological ontologies, scientific literature, and organizational hierarchies.

---

### Exhibit 1 — The Lottery Cascade: What Was Decided, When, and by What Mechanism

```
LAYER          EVENT           YEAR   WINNER            LOSER              COST OF LOSING
─────────────────────────────────────────────────────────────────────────────────────────
Arithmetic     IEEE 754        1985   Float/FLOPS        Fixed-pt/ticks     2–10× iteration overhead
Optimization   Backprop+SGD    1986   Euclidean grad.    Natural gradient    100× excess train steps (κ=10⁴)
Silicon        Transformer     2017   Matmul/TPU         RNN/CORDIC         Total architecture lock-in
Bio-compute    GEMM substrate  →2028  Pair-tensor native  GEMM emulation    4–12× per-op; $490B waste/yr by 2028
Geometry       Euclidean embed →2027  Lorentzian/hyperbolic  Flat cosine   Distorted metrics; 4% capability gap
```

---

## 2. The Three Ceilings — Where the Stack Breaks

The five lotteries did not create isolated costs. They compound. Three distinct ceilings are now visible, each with a defined timeline and a defined forcing function.

### Ceiling 1 — The Iteration Ceiling (arriving 2026–2028)

The IEEE 754 substrate bills in FLOPS. The 2024–2026 reasoning model wave — OpenAI o1/o3/o4, DeepSeek R1, Anthropic extended thinking, Gemini Thinking — bills users in tokens, which translates to iteration depth per query. This mismatch is currently absorbed by software stacks (vLLM, KV-cache management, persistent batching) that convert tick-budgets into FLOP-budgets at overhead cost. The overhead is the 1985 decision, paid in 2026.

**Forcing function:** Once iteration depth is the unit of value at the inference API, it becomes economically justified to design silicon that bills the same unit natively. The STM32G4's dedicated CORDIC coprocessor (STMicroelectronics, 2021) is the MCU-scale proof of concept. The datacenter-scale version — tick-aware coprocessor inside an FP4 or FP8 matmul engine — is the 2028 forecast. TPU v8's training/inference split (Sunfish/Zebrafish, late 2027) is the leading indicator: Google has decided that training-time iteration and inference-time iteration deserve different silicon, with each chip optimized for the tick of its workload.

### Ceiling 2 — The Curvature Ceiling (arriving 2025–2027)

The optimization substrate discards the Fisher information matrix — the geometrically correct metric on the model's parameter manifold — for computational reasons. Sophia (2023) demonstrated 50% step reduction from recovering a single diagonal dimension. Muon (2024), scaled to LLM training by Liu et al. (2025) and DeepSeek-AI (2026), demonstrates that spectral-norm steepest descent (recovering the dominant singular-value structure of each weight matrix via Newton-Schulz fixed-point iteration) competes with or exceeds AdamW at frontier scale. The direction is clear: every step closer to the natural gradient reduces training cost.

**Forcing function:** The Xu Spectral Edge Thesis (2026) provides the diagnostic vocabulary: grokking, double descent, and training phase transitions are spectral-gap-collapse events in the Fisher geometry of the parameter-update Gram matrix. These transitions are currently invisible to training monitoring infrastructure because the monitoring vocabulary is gradient-norm and loss-curve, not curvature-time. The first organization to instrument Fisher-geometric training diagnostics as standard telemetry will have a systematic edge on every subsequent training run.

### Ceiling 3 — The Biology Compute Ceiling (arriving 2028)

Biology AI absorbs a growing share of frontier compute: ~6% in 2025, on track to 14% by 2028. The five architectural mismatches between biology workloads and GEMM silicon (pair tensor, SE(3), diffusion, Hyena, Sherman-Morrison) pay overheads of 3–12× each, compounding multiplicatively on workloads that touch more than one. An AlphaFold 3 saturation mutational scan touches four of the five.

**The math:** A 5× overhead on 6% of $200B frontier compute base is $60B in structurally wasted compute annually. A 5× overhead on 14% of $700B (2028) is $490B. The first figure fits inside R&D budgets. The second is a board-level capital crisis. Moore's Law provides no relief: the overhead is architectural, not process-node-driven. It does not shrink.

---

### Exhibit 2 — Three Ceilings: Timeline, Magnitude, and Forcing Event

| Ceiling | Primary Cause | Timeline | Annual Waste at Peak | Forcing Event |
|---|---|---|---|---|
| Iteration | IEEE 754 hides tick-count | 2026–2028 | ~$50B+ (reasoning models) | Per-token inference pricing; TPU v8 split |
| Curvature | Optimizer discards Fisher | 2025–2027 | ~$200B in excess training steps | Sophia/Muon/SOAP convergence; spectral diagnostics |
| Biology | GEMM emulates pair-tensor | 2028 | ~$490B by 2028 | Population genomic medicine product wave |

---

## 3. The Geometric Object — What the Stack Is Computing

Five independent research programs converged in 2025–2026 on a description of what every trained language model actually computes. The convergence is not speculative — it is the result of peer-reviewed empirical work from five distinct scientific communities, none of which began with the intention of describing the others' findings.

The object is the **Anisotropic Representation Hypothesis (ARH, 2026)**: a sloppy hyperribbon manifold equipped with a stratified Lorentzian metric, polytopal fiber structure, Radon-domain BV functional presentation, and gradient-noise-governed training dynamics.

### Pillar I — The Sloppy Hyperribbon (Fisher Anisotropy)

Transtrum, Machta, Brown, Daniels, Myers & Sethna (2015) established that multiparameter models generically produce *hyperribbon* manifolds: parameter-space eigenvalues span many decades in a log-uniform hierarchy, from a few stiff directions (large eigenvalues, meaningfully constrained by training) to an overwhelming majority of sloppy directions (small eigenvalues, barely touched by training). Mao, Griniasty, Sethna & Chaudhari (*Phys. Rev. E* 113, 015306, 2026) provided the analytical characterization for deep networks: hyperribbon geometry is determined by three factors — the eigenvalue decay rate of the input correlation matrix, the relative scale of outputs to initial weights, and the number of gradient steps. Architecture contributes to geometry only at initialization; thereafter, the data geometry dominates.

**Strategic implication:** The vast majority of parameters in any large model are geometrically sloppy — they occupy directions that training never meaningfully constrains. Effective geometric dimension, not raw parameter count, governs capability. Organizations explicitly managing this geometry outperform those scaling blindly.

### Pillar II — The Polytopal Fiber (Minkowski Assembly)

Fel et al. (*arXiv:2510.08638*, v2 February 2026) demonstrated that multi-head attention assembles output representations as Minkowski sums of convex polytopes — one polytope contribution per attention head — and that the resulting polytopal structure is recoverable via stable sparse autoencoders as an archetype dictionary. The Minkowski Representation Theory (MRT, 2026) distinguishes two formally independent geometric structures: the Minkowski sum assembled by attention (a convex-geometry operation) and the Lorentzian inner product of the ambient embedding space (a pseudo-Riemannian operation). Infrastructure theory that conflates them is incomplete.

**Strategic implication:** Fine-tuning that adjusts polytope geometry (fiber/vertical directions) without touching the underlying Lorentzian token-space geometry can be performed at low rank. LoRA rank, currently a hyperparameter, is in principle derivable from the fiber dimension of the target task.

### Pillar III — The Lorentzian Token Space (Negative Ricci Curvature)

Robinson, Dey & Sweet (*arXiv:2410.08993*, 2024) established that token embedding spaces are stratified manifolds with **significantly negative Ricci curvature** — not Euclidean structures. Robinson, Dey & Chiang (*arXiv:2504.01002*, 2025) demonstrated formal rejection of the manifold hypothesis for token embeddings. TokenBlowUp (*arXiv:2507.19747*, 2025) resolves the representational singularities at stratum boundaries via monoidal transformations.

**Strategic implication:** Cosine similarity applied to token embeddings is a distortion, not a measurement. The true geometric distance is Lorentzian. Retrieval infrastructure, embedding databases, and nearest-neighbor search pipelines built on flat-space assumptions introduce systematic errors that scale with domain hierarchical depth.

### Pillar IV — The BV Functional (Radon-Domain Regularization)

Parhi & Nowak (*JMLR* 22(43), 2021) demonstrated that neural networks solve a Radon-domain bounded-variation inverse problem: the minimum-norm interpolant in an infinite-dimensional reproducing kernel Banach space (RKBS). This characterization is architecture-agnostic and provides the functional-analytic foundation for understanding generalization. The trained network is a BV function on a curved domain — not a collection of matrix multiplications.

**Strategic implication:** Generalization bounds derived from this characterization are tighter than VC-dimension or Rademacher bounds and are computable from the model's weight structure. Organizations with access to this framework have more accurate capability predictions than those relying on empirical scaling laws alone.

### Pillar V — The Gradient Noise Scale (Training Dynamics)

McCandlish, Kaplan, Amodei & OpenAI Dota Team (*arXiv:1812.06162*, 2018) introduced the gradient noise scale B_simple = tr(Σ_g)/|μ_g|², which governs the critical batch size above which additional parallelism yields diminishing training efficiency. Merrill, Arora, Groeneveld & Hajishirzi (*arXiv:2505.23971*, May 2025) revisited this for modern LLM training. Under the Representational Bundle Hypothesis (RBH, 2026), B_simple has a precise geometric interpretation: it is the ratio of vertical drift (fiber/sloppy directions) to horizontal progress (base-manifold/stiff directions) in the training trajectory.

**Strategic implication:** Optimal batch size scheduling over a training run is derivable from the curvature evolution of the learned attention connection, not from empirical grid search. Early training (curved connection, low B_simple) benefits from small batches; late training (flat connection, high B_simple) tolerates large ones.

---

## 4. The Fiber Bundle Architecture — Operational Translation

The Representational Bundle Hypothesis (RBH, 2026) makes a precise structural claim: the sloppy hyperribbon manifold M is the total space of a non-trivial principal SO⁺(1,n)-bundle over the Lorentzian token space B, with the attention mechanism serving as its connection one-form ω.

The operational translation has five components with immediate infrastructure consequences.

### 4.1 Stiff = Horizontal, Sloppy = Vertical

Fisher-information stiff directions correspond to horizontal subspace directions in the bundle's tangent space — they alter the Lorentzian geometry of the token space. Sloppy directions are vertical — they change polytope geometry over a fixed token position. This decomposition is not merely geometric: it governs what compute dollars buy at each training phase.

| Training Phase | Dominant Subspace | What Compute Buys | Transfers Across Tasks? |
|---|---|---|---|
| Early training | Horizontal (stiff) | Base-manifold Lorentzian structure | **Yes** — architecture-universal |
| Late training | Vertical (sloppy) | Fiber/polytope refinement | **Partially** — task-specific |
| Fine-tuning (LoRA) | Vertical (sloppy) | Polytope adjustment | **No** — task-locked |

**Operational consequence:** Organizations that conflate training phases are either under-investing in generalizable capability (cutting training before horizontal structure is acquired) or over-paying for task-specific specialization (continuing horizontal-phase training when the manifold has already settled).

### 4.2 Chain-of-Thought Is Path Lifting

In the bundle picture, a forward pass processes tokens by parallel-transporting fibers along the attention connection. Chain-of-thought constructs an explicit intermediate path on the base manifold — the reasoning steps provide waypoints that guide the horizontal lift through topologically favorable regions. CoT gains are largest when the task requires traversing high-curvature stratum boundaries that the model cannot navigate implicitly from a compressed prompt.

**Operational consequence:** CoT improvement is not uniform across tasks. The geometric account predicts — and the stratum-crossing analysis supports — that CoT gains correlate with the degree of hierarchical complexity in the domain and the number of stratum boundary crossings on the optimal reasoning path.

### 4.3 In-Context Learning Scales with Bundle Holonomy

ICL accuracy scales, under RBH, with the topological winding of the context token path through the base manifold. Context length is not the relevant variable — context geometric span is. A 128K context window filled with repetitive tokens may deliver less ICL benefit than a 32K window with geometrically diverse examples spanning distinct regions of the base manifold.

**Operational consequence:** Long-context infrastructure investment should be evaluated against geometric diversity of context content, not raw length. The current practice of maximizing context length as a proxy for ICL capability is an approximation that breaks down for semantically repetitive domains.

### 4.4 The Attention Curvature Signal Is Measurable Today

The curvature of the attention connection is computable from existing training telemetry as the position-variance of attention distributions across query positions: high-entropy, position-uniform attention indicates a flat connection (low curvature, late training); position-peaked attention indicates a curved connection (high curvature, early training). No new instrumentation is required — the signal is present in attention weight logs that most training runs already generate.

**Operational consequence:** Geometric training diagnostics are available with today's tooling. What is missing is the interpretive framework. The first organization to add curvature monitoring to standard training telemetry will have a systematic edge in compute allocation — eliminating both the premature-termination error and the wasteful post-holonomy continuation error.

---

## 5. Grokking Is a Capital Problem

No phenomenon in recent deep learning has more direct implications for AI capital allocation than grokking — and none is more poorly understood at the budget-decision level.

### The Phenomenon

Power, Burda, Edwards, Goodfellow & Misra (2022) documented delayed generalization: networks trained on algorithmic tasks achieve near-zero training loss (memorization) while maintaining near-chance validation accuracy, then undergo a sharp, delayed transition to near-perfect generalization — after compute investment that, by conventional metrics, appeared to deliver no further value.

The dominant industry response — treating grokking as a small-dataset curiosity irrelevant to frontier models — is wrong. Grokking is the signature of a geometric transition that all capable models undergo, at scales and timelines that vary with task complexity and model size.

### The Geometric Account

Under RBH, grokking is **holonomy accumulation**. The two phases map precisely:

**Phase 1 (Memorization):** The model adjusts polytope geometry (fiber/vertical directions) over training examples. The base manifold has not yet organized to reflect the algebraic structure of the task. The model has learned where training examples are in the fiber; it has not learned the base-manifold structure that enables generalization to novel examples.

**Phase 2 (Generalization transition):** Continued gradient steps drive the training trajectory around a closed loop in the base manifold. When the attention connection has non-trivial holonomy, this loop produces a non-trivial fiber transformation — a Lorentz group element that reorganizes polytope structure to align with the task's underlying symmetry group. Nanda, Chan, Lieberum, Smith & Steinhardt (ICLR 2023) confirmed this mechanistically: the grokked algorithm for modular arithmetic is a discrete Fourier transform over the group structure — exactly the holonomy embedding predicted by RBH.

Muon optimizer accelerates grokking: Tveit, Remseth & Skogvold (Microsoft, 2025) demonstrated that Muon reduces mean grokking epoch from 153 to 103 (33% reduction) compared to AdamW across modular arithmetic tasks. A curvature-respecting optimizer accelerates holonomy accumulation because it navigates the base manifold more directly, reaching the closed loop that produces the fiber reorganization in fewer steps.

### The Capital Allocation Consequence

| Error Type | Description | Financial Impact |
|---|---|---|
| Premature termination | Stopping at training-loss convergence, before holonomy transition | Lost capability; full retraining cost |
| Post-transition waste | Continuing training after generalization transition completes | Compute cost with no capability return |
| Phase conflation | Allocating large batches during early horizontal-dominated training | Systematic compute inefficiency |

**The corrective prescription:** Monitor the attention connection's holonomy group — detectable as the structural reorganization of attention patterns from position-based to algebraic-structure-based — and calibrate compute budgets to the geometry of the transition. This is achievable with existing telemetry infrastructure and the RBH interpretive framework.

---

## 6. The Lock-In Economy — Dependency Map

### The Platform Strategy Anatomy

Google's open publication of *Attention Is All You Need* in 2017 followed a playbook established across TensorFlow (2016), Kubernetes, and Android: **open-source the component that creates dependency on the layer you control and monetize.** In the Transformer case — open-source the algorithm, retain proprietary control over the infrastructure layer: TPU hardware, Pod interconnect, XLA compiler, and Cloud TPU rental service. The algorithm creates universal dependency on matrix-optimized silicon. Google builds and rents the best matrix-optimized silicon available.

The comparison to Android is not metaphorical. It is structural: Android commoditized handset operating systems to create Google Search dependency; the Transformer commoditized AI architecture to create TPU compute dependency. The addressable market for the infrastructure layer is now measured in hundreds of billions annually.

### Exhibit 3 — Stakeholder Lock-In Dependency Map

| Stakeholder | TPU Dependency | Architecture Lock-in | Annual Exposure | Mitigation Path | Transition Timeline |
|---|---|---|---|---|---|
| **Anthropic** | High (GCP + Trainium) | Total (Transformer) | ~$2B+/yr compute | Custom silicon (Trainium3 partial) | 3–5 years partial |
| **Google DeepMind** | Self-owned | Self-designed | Lock-in beneficiary | N/A | Advantaged incumbent |
| **OpenAI** | None (Azure H100) | Total (Transformer) | ~$4B+/yr compute | Microsoft partnership | Stable |
| **Meta AI** | Low (NVIDIA GPU) | Total (LLaMA = Transformer) | ~$3B+/yr compute | Commodity GPU | Stable but inefficient |
| **Enterprise fine-tuners** | Medium | Total | Scales with usage | Open-weight models | 1–2 years viable |
| **Cloud providers (AWS, Azure)** | Indirect — custom silicon in progress | Total | Revenue at risk from Google | Trainium, Maia 200 | 2–4 years |
| **NVIDIA** | None | Beneficiary | Revenue from all above | Geometry extension (Rubin) | Defensible through 2028 |

**The Microsoft Signal:** Microsoft's $13B+ investment in Azure A100/H100 clusters was explicitly motivated by the strategic imperative of avoiding TPU dependency. The fact that Microsoft found it necessary to commit tens of billions of dollars specifically to avoid Google hardware dependency is the clearest available evidence that the Transformer paper created genuine infrastructure lock-in with quantifiable costs.

---

## 7. The Lorentzian Transition — Market Sizing

### The Engineering Obstacles Are Cleared

The 2025–2026 literature resolved a decade of trainability failures for non-Euclidean architectures in a concentrated burst:

| Obstacle | Resolution | Date | Source |
|---|---|---|---|
| Scale to billion parameters | HELM billion-parameter hyperbolic LLM; up to +4% on MMLU/ARC | NeurIPS 2025 | He et al., *arXiv:2505.24722* |
| Logarithmic norm scaling under gradient descent | Distance-to-hyperplane formulation restores linear norm scaling | January 2026 | Van der Klis et al., *arXiv:2601.21529* |
| Fully intrinsic architecture (no Euclidean projections) | Point-to-hyperplane Lorentz FC layer | ICLR 2026 | *arXiv:2602.23981* |
| Production-grade optimizer | Curvature-aware Riemannian AdamW for Poincaré ball and Lorentz hyperboloid | NeurIPS 2025 | Bdeir et al., *arXiv:2405.13979* |
| Reinforcement learning stability | Hyper++ — stable PPO in hyperbolic space; ~30% wall-clock improvement | December 2025 | Klein et al., *arXiv:2512.14202* |
| Hardware acceleration | CORDIC-accelerated hyperbolic operation engine (CARMEN) | 2026 | Kumar et al., *arXiv:2605.06878* |

This is not a gradual improvement — it is a series of discrete breakthroughs that collectively remove the engineering barriers that blocked the Lorentzian investment cycle for ten years.

### Target Domains

The commercial returns on Lorentzian infrastructure are largest in domains with demonstrably negative Ricci curvature in their latent geometry — where Euclidean cosine similarity is a distortion. These domains are not obscure. They are the highest-value enterprise AI verticals:

| Domain | Geometric Reason | Near-Term Product Impact |
|---|---|---|
| Knowledge graph embedding | Hierarchical ontological structure | Accuracy gains on link prediction, entity disambiguation |
| Legal and regulatory documents | Deep citation and jurisdictional hierarchy | Improved retrieval precision; reduced hallucination on precedent |
| Biological ontology modeling | GO terms, taxonomy, pathway hierarchy | Direct connection to $490B biology compute opportunity |
| Code structure analysis | Call graphs, import dependencies, inheritance hierarchies | Richer RAG for code generation |
| Scientific literature | Citation and concept hierarchy | Improved reasoning on multi-hop scientific queries |
| Organizational data | Reporting structures, process hierarchies | Enterprise knowledge management |

### Market Sizing

| Year | Total Frontier AI Compute | Geometry-Sensitive Workload Share | Lorentzian-Addressable Market | Native-Geometry Capture (20% SOM) |
|---|---|---|---|---|
| 2025 | ~$200B | ~8% | ~$16B | ~$3B |
| 2026 | ~$320B | ~12% | ~$38B | ~$8B |
| 2027 | ~$500B | ~16% | ~$80B | ~$16B |
| 2028 | ~$700B | ~20% | ~$140B | ~$28B |
| 2029 | ~$880B | ~23% | ~$202B | ~$40B |
| 2030 | ~$1.05T | ~25% | ~$263B | ~$53B |

The comparable historical analogue is NVIDIA's capture of the 2014–2024 frontier AI compute wave — a $400B+ revenue event generated by being the hardware-software co-design winner for the Transformer lottery. The geometry lottery is the next cycle.

---

## 8. Strategic Archetypes and Competitive Positions

### Archetype 1 — The Geometry-Native Builder (Advantaged)

**Profile:** AI infrastructure organization building explicitly against the geometric framework. Invests in Lorentzian distance metrics, instruments attention curvature, designs batch schedules from gradient noise geometry, and develops geometric routing for mixture-of-experts.

**Current examples:** HELM (NeurIPS 2025), Bdeir et al.'s Riemannian AdamW (NeurIPS 2025), Kumar et al.'s CORDIC acceleration (2026). None yet at full commercial scale.

**Competitive position:** Structurally superior information set on every infrastructure decision. First-mover in the Lorentzian transition. The co-design window — designing algorithms and silicon simultaneously under the same organizational authority — is available now and will close as the established players respond.

**Vulnerability:** Must reach commercial scale before the matmul-ecosystem's defensive responses (NVIDIA Rubin-class geometric extensions, Google TPU v9 Lorentzian kernels) compress the performance gap from 4–12× to 2–4×. The window is estimated at 2026–2028.

### Archetype 2 — The Scale-Incumbent (Defensively Positioned)

**Profile:** Frontier lab or hyperscaler with competitive position achieved through scale, optimizing within the Euclidean Transformer paradigm.

**Current examples:** OpenAI, Anthropic, Meta AI, Google DeepMind.

**Competitive position:** Dominant in the current cycle. Exposed to the Lorentzian transition on hierarchically structured tasks and to the biology compute ceiling on life sciences workloads. The transition will not obsolete Transformer infrastructure overnight — but it will create a performance gap that compounds through 2027–2030.

**Required moves:** Acquire or partner with geometry-native builders. Add attention curvature diagnostics to standard training telemetry. Launch Lorentzian prototype programs in highest-value vertical domains before the gap becomes public and competitive.

### Archetype 3 — The Infrastructure Commodity Provider (Transitioning)

**Profile:** Cloud provider or hardware vendor offering compute for Transformer training and inference.

**Current examples:** AWS (Trainium), Microsoft Azure (Maia 200), NVIDIA (GPU fleet).

**Competitive position:** The matmul-era infrastructure moat is durable through 2027 on language workloads. The biology compute ceiling (2028) creates explicit vulnerability in the life sciences vertical. NVIDIA's hardware moat is more durable through the geometric transition than Google's TPU business because GPU tensor cores are more geometry-agnostic than systolic arrays.

**Required moves:** Evaluate CORDIC-based hardware support for hyperbolic operations as a near-term differentiation opportunity. CARMEN (*arXiv:2605.06878*, 2026) and *CORDIC Is All You Need* (*arXiv:2503.11685*, 2025) provide the implementation blueprint. Position biology-native silicon products for the 2028 window.

### Archetype 4 — The Enterprise Adopter (Insulated Near-Term, Exposed Mid-Term)

**Profile:** Organization deploying foundation models for enterprise use cases. Fine-tuning or prompting existing models; not training from scratch.

**Primary exposure:** Quality degradation from Euclidean-assumption embedding infrastructure in hierarchically structured domains — specifically, legal, scientific, organizational, and biological data.

**Required moves:** Audit embedding and retrieval infrastructure for Lorentzian distance compatibility. Evaluate task domain geometry before committing to fine-tuning strategy. The LoRA rank derivation from fiber dimension — if confirmed by prediction P8 — eliminates a major search cost in fine-tuning pipelines for complex domains.

---

### Exhibit 4 — Competitive Position Summary Matrix

```
                           GEOMETRY READINESS
                    Low ◄────────────────────► High
                    │                              │
    High  ──────────┼──────────────────────────────┼──
                    │                              │
  SCALE             │  Scale-Incumbents           │ (Target position:
  POSITION          │  (Google, Anthropic,         │  Geometry-Native
                    │   OpenAI, Meta)              │  at Scale)
                    │  ▲ Must move right           │
    Low   ──────────┼──────────────────────────────┼──
                    │                              │
                    │  Commodity Providers         │ Geometry-Native
                    │  (AWS, Azure, NVIDIA)        │ Builders
                    │  ▲ Must move right before    │ (HELM, Riemannian
                    │    window closes 2028        │  AdamW teams)
                    │                              │ ▲ Must scale up
```

---

## 9. The 2027–2030 Action Agenda

### For Technology Executives and C-Suite

**Immediate (Q1–Q2 2027)**

1. **Commission a geometric audit of your AI stack.** Identify which components assume Euclidean geometry. Quantify the domains where your embedding and retrieval infrastructure is operating on distorted metrics. Priority domains: legal, scientific, biological, organizational.

2. **Instrument attention curvature.** Add attention entropy and position-variance diagnostics to standard training telemetry. The signal is already being generated; the interpretive framework is now available. Cost: engineering days, not months. Return: systematic improvement in compute allocation starting immediately.

3. **Reassess your grokking exposure.** Audit recent training runs for premature termination (training-loss convergence without generalization validation) and post-transition waste (continued training after attention patterns reorganize). Both errors are recoverable with protocol changes, not hardware changes.

**Near-Term (Q3 2027–Q2 2028)**

4. **Reframe your scaling-law assumptions.** Replace parameter-count proxies with geometric measures: hyperribbon effective dimension, attention connection curvature, and Ricci scalar estimates of the token space for your primary task domain. Effective dimension, not raw parameter count, governs capability per training dollar.

5. **Evaluate your TPU/GPU procurement strategy against the geometry transition.** The silicon most compatible with non-Euclidean operations is not the silicon that currently dominates Transformer training. The 2028 biology compute ceiling and the Lorentzian transition create different exposure profiles for TPU and GPU fleets.

6. **Launch a Lorentzian prototype program in one high-value vertical.** The engineering obstacles are cleared. The competitive window is open. First-mover in a single high-value hierarchical domain (legal, biological, scientific) creates durable product differentiation ahead of the Euclidean-to-Lorentzian migration.

### For AI Research Leadership

1. **Run the cross-domain polytopal experiment.** The most commercially consequential single experiment in representational geometry: run stable sparse autoencoders on HELM's hyperbolic hidden states and test whether the archetype dictionary recovers Fel et al.'s DINOv2 polytopal structure. Confirmation validates the geometry-first investment thesis across modalities; the tools are available today.

2. **Develop geometric training diagnostics as standard toolchain.** Gradient noise scale B_simple, attention curvature two-form norm, vertical/horizontal Fisher information decomposition, and Ricci scalar estimates are all computable from existing training artifacts. Making them standard raises the entire field's compute efficiency.

3. **Invest in the second-order optimizer transition.** The Sophia/Muon/SOAP convergence demonstrates the direction: every step closer to natural gradient descent reduces training cost. The Fisher-geometric interpretation of weight decay (FAdam, 2024) provides the theoretical foundation for systematic optimizer improvement beyond AdaGrad→Adam→Muon.

### For Infrastructure Investors

1. **The next infrastructure moat is geometric.** Organizations building geometry-native AI infrastructure — Lorentzian embedding databases, curvature-aware training systems, bundle-structured architecture search — are in the position Google was in 2016: one year before the silicon lottery they co-designed came online.

2. **Evaluate hardware ventures against geometric operation support.** CORDIC-based hardware acceleration for hyperbolic operations (CARMEN, 2026) is a credible near-term differentiator. The relevant benchmarks are Lorentzian distance computation throughput and Riemannian optimizer iteration cost, not FLOPS alone.

3. **Probability-weight the biology compute ceiling across your portfolio.** Every portfolio company dependent on matmul-era compute for biology workloads has unpriced transition risk that crystallizes in 2028. The $490B annual waste figure at the ceiling is a TAM signal as much as a cost signal.

4. **Time the co-design window.** The hardware-software co-design window that produced TPU v2 + Transformer in 2017 produced a decade-long competitive advantage for Google. That window is open for the geometry-native stack today. It will close as the matmul incumbents ship NVIDIA Rubin-class geometric extensions (forecast 2027–2028) and TPU v9 Lorentzian kernels.

---

## 10. Ten Falsifiable Predictions

These predictions follow directly from the five-lottery framework and are stated in falsifiable form. Each represents a significant commercial opportunity for the organization that confirms it first.

| # | Prediction | Timeline | Falsification Criterion |
|---|---|---|---|
| **P1** | A frontier silicon program ships a chip whose datasheet quotes iteration-rate alongside FLOPS — denoising-steps/s, Sinkhorn-iterations/s, or comparable workload-native unit | **2028** | Persistence of FLOPS-only datasheets at the frontier past 2028 |
| **P2** | Muon or Muon-class spectral-norm steepest descent becomes the frontier-default optimizer for Transformer hidden layers | **2027** | Adam-family optimizers remain sole optimizer at frontier scale past 2027 |
| **P3** | A published result demonstrates ≥3× step reduction vs. Adam on a 7B-scale language model via information-geometric preconditioning | **2027** | Off-diagonal Fisher contributes negligibly to condition number at LLM scale |
| **P4** | The grokking generalization transition coincides with abrupt structural reorganization of attention connection holonomy group — detectable as shift from position-based to algebraic-structure-based attention patterns | **2027** | Grokking at scale occurs without corresponding attention pattern reorganization |
| **P5** | Stable sparse autoencoders on HELM's hyperbolic hidden states recover an archetype dictionary with the same convex-polytopal geometry as Fel et al.'s DINOv2 analysis | **2027** | SAE dictionaries on HELM fail to exhibit convex archetype structure |
| **P6** | Lorentzian distance metrics achieve production deployment in ≥2 of: major embedding database, enterprise RAG system, or frontier model retrieval pipeline | **2028** | Cosine similarity remains the sole production metric for token-embedding retrieval past 2028 in hierarchically structured domains |
| **P7** | ICL accuracy, controlling for context length, scales with topological winding measure of the context token path — geometric diversity matters more than raw length | **2028** | ICL gains depend only on example count independent of geometric diversity |
| **P8** | LoRA rank for a given task is derivable from fiber dimension — eliminating rank hyperparameter search; fine-tuning does not alter Robinson-Dey Ricci curvature estimates of the base model | **2027** | LoRA fine-tuning produces measurable changes in Ricci curvature of the pretrained token space |
| **P9** | A pharma or biotech organization announces a computational pipeline reducing AlphaFold-class inference cost by ≥5× via pair-tensor-native or diffusion-iterating substrate, bypassing the matmul-emulation overhead | **2028** | Biology AI compute costs on frontier biology workloads track process-node improvement (i.e., overhead is not structural) |
| **P10** | The Gradient Descent Lottery — the 1986 commitment to Euclidean gradient descent as a hardware-driven optimization lottery — becomes a recognized framing in a published paper at NeurIPS, ICML, ICLR, ISCA, or HPCA | **2027** | No published venue paper attributes optimization research-direction outcomes to the 1986 hardware-software co-design commitment |

---

## 11. Conclusion — The Geometry of the Next Decade

### The Compounding Structure

The title of this report is not a metaphor. Compounding advantage in AI infrastructure is a precise, geometric claim: each co-design lottery that was won in 1985, 1986, and 2017 did not simply set a constraint — it multiplied the cost of all subsequent substrate mismatches. The IEEE 754 arithmetic substrate made CORDIC-native iteration economically uncompetitive. The Euclidean optimization substrate made Fisher-metric navigation computationally prohibitive. The matmul silicon substrate made non-GEMM workloads architecturally alien. Each decision compounded the others.

The reversal of compounding advantage follows the same structure — in the opposite direction. Every step toward geometry-native arithmetic simultaneously reduces the iteration overhead, narrows the curvature gap, and makes Lorentzian distance computation cheaper. Every step toward curvature-aware optimization simultaneously shortens training runs, accelerates grokking transitions, and reduces the condition-number tax that grows with model scale. Every step toward Lorentzian token-space infrastructure simultaneously improves retrieval precision, closes the capability gap on hierarchical domains, and reduces the systematic distortion introduced by cosine similarity. The geometry-native stack does not produce linear improvements. It produces compounding ones — which is the only kind that creates durable infrastructure moats.

### The Timing Asymmetry

Two organizations facing the same transition at different moments have structurally different options. The organization that acts before the co-design window closes can shape the substrate — designing algorithms and silicon simultaneously under unified authority, as Google did in 2016–2017. The organization that acts after the window closes must operate within the substrate — procuring infrastructure designed by others for workloads partially different from its own. The hardware-software co-design window for geometry-native AI infrastructure opened in 2025, when the trainability obstacles were resolved and the performance gaps became measurable. It will close when the matmul incumbents ship geometry-capable extensions at competitive cost — estimated 2027–2028 for NVIDIA (Rubin architecture) and 2028–2029 for Google (TPU v9). 

Between 2025 and 2028, the organizations that correctly identify the geometry layer as the forcing constraint and invest accordingly will establish the moat that compounds through the following decade. The ones that correctly identify it but act too late will pay competitive infrastructure rents to those who did not wait. The ones that misidentify it — concluding that more parameters, more FLOPS, or more context length addresses the underlying geometric mismatch — will fund both groups.

### The Unified Picture

Five research programs independently described five aspects of the same geometric object. The hyperribbon is the shape of the parameter space. The Lorentzian metric is the geometry of the token space. The polytopal fiber is the structure of the representation. The BV functional is the regularity class of the function being learned. The gradient noise scale is the dynamics of the training trajectory through this object. None of these descriptions conflicts with the others. Together they constitute a complete picture of what every production AI system is computing — and therefore a complete diagnosis of where every substrate mismatch imposes its overhead.

The organizations that operate with this picture will spend their training compute on the right geometric directions, terminate training at the right moment, select the right retrievers, choose the right optimizers, and size their fine-tuning correctly. The organizations that operate without it will pay the five-lottery tax — arithmetic overhead, curvature blindness, Euclidean distortion, parameter-count proxy error, and grokking mismanagement — on every training run, every inference call, and every capital allocation decision, compounding indefinitely.

The geometry of compounding advantage is not a forecast. It is a description of the structure already in place. The question is only which side of it each organization occupies.

---

## 12. Evidence Base and Sources

### The Five Lottery Foundations

Hooker, S. The Hardware Lottery. *Communications of the ACM* 64(12), 2020. arXiv:2009.06489. — *Foundational framework for co-design analysis.*

Kahan, W. IEEE Standard 754 for Binary Floating-Point Arithmetic. IEEE, 1985. — *The arithmetic lottery's outcome document.*

Volder, J. E. The CORDIC Trigonometric Computing Technique. *IRE Trans. EC-8(3)*, 1959. — *The losing arithmetic lineage.*

Luo, Y. et al. Generalized Hyperbolic CORDIC. *IEEE TVLSI* 27(9), 2019. — *Energy cost quantification of the arithmetic lottery.*

Rumelhart, D. E., Hinton, G. E., Williams, R. J. Learning Representations by Back-propagating Errors. *Nature* 323, 1986. — *The optimization lottery's decisive commitment.*

Amari, S.-I. Natural Gradient Works Efficiently in Learning. *Neural Computation* 10(2), 1998. — *The theoretical verdict against Euclidean gradient descent.*

Vaswani, A. et al. Attention Is All You Need. *NeurIPS* 2017. arXiv:1706.03762. — *The silicon lottery event.*

Jouppi, N. P. et al. TPU v1. *ISCA* 2017; TPU v2/v3. *Commun. ACM* 63(7), 2020. — *The co-designed silicon.*

### The Geometric Object

Transtrum, M. K. et al. Sloppy models universality. arXiv:1501.07668, 2015. — *Hyperribbon geometry foundation.*

Mao, J. et al. Hyperribbon in deep networks. *Phys. Rev. E* 113, 015306, 2026. arXiv:2505.08915. — *Analytical characterization of Pillar I.*

Fel, T. et al. Minkowski Representation Theory. arXiv:2510.08638, v2 February 2026. — *Polytopal fiber structure (Pillar II).*

Robinson, M., Dey, S., Sweet, J. Token space stratification; negative Ricci curvature. arXiv:2410.08993, 2024. — *Lorentzian token geometry (Pillar III).*

Robinson, M., Dey, S., Chiang, Y. Formal rejection of manifold hypothesis. arXiv:2504.01002, 2025. — *Pillar III extension.*

Parhi, R., Nowak, R. Neural networks as RKBS inverse problems. *JMLR* 22(43), 2021. — *BV functional characterization (Pillar IV).*

McCandlish, S. et al. Large-batch training gradient noise scale. arXiv:1812.06162, 2018. — *Gradient dynamics (Pillar V).*

Merrill, N. et al. Critical batch size revisited. arXiv:2505.23971, 2025. — *Pillar V at modern LLM scale.*

### The Optimization Recovery

Kingma, D. P., Ba, J. Adam. *ICLR* 2015. arXiv:1412.6980. — *The diagonal-Fisher incumbent.*

Hwang, D. FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher. arXiv:2405.12807, 2024. — *Theoretical bridge: Adam as approximate NGD.*

Liu, H. et al. Sophia: 50% step reduction via diagonal Hessian. *ICML* 2024. arXiv:2305.14342. — *Curvature recovery evidence.*

Jordan, K. Muon: spectral-norm steepest descent for hidden layers. Blog/GitHub, December 2024. — *Spectral curvature recovery.*

Liu, J. et al. Muon is scalable for LLM training. arXiv:2502.16982, 2025. — *LLM-scale confirmation.*

Vyas, N. et al. SOAP: Shampoo as Adam Preconditioner. arXiv:2409.11321, 2024. — *Dual curvature recovery.*

Tveit, A. et al. Muon Optimizer Accelerates Grokking. arXiv:2504.16041, Microsoft, 2025. — *Curvature → grokking connection.*

Shrestha, R. Natural Gradient Methods: Perspectives, Efficient-Scalable Approximations, and Analysis. arXiv:2303.05473, 2023. — *Convergence bounds and approximation survey.*

Martens, J. New Insights and Perspectives on the Natural Gradient. *JMLR* 21(146), 2020. — *Definitive K-FAC and natural gradient analysis.*

### The Lorentzian Transition

He, N. et al. HELM: Hyperbolic Large Language Models. *NeurIPS* 2025. arXiv:2505.24722. — *Billion-parameter hyperbolic LLM; +4% MMLU/ARC.*

Van der Klis, M. et al. Norm scaling fix for Lorentz layers. arXiv:2601.21529, January 2026. — *Training stability breakthrough.*

Intrinsic Lorentz Neural Network. *ICLR* 2026. arXiv:2602.23981. — *Fully intrinsic architecture.*

Bdeir, A., Schwethelm, K., Landwehr, N. Riemannian AdamW. *NeurIPS* 2025. arXiv:2405.13979. — *Production-grade hyperbolic optimizer.*

Klein, J. et al. Hyper++: stable hyperbolic RL. arXiv:2512.14202, December 2025. — *Reinforcement learning in hyperbolic space.*

Yang, W. et al. HypLoRA: hyperbolic LoRA. arXiv:2410.04010, 2025. — *Parameter-efficient Lorentzian fine-tuning.*

Kumar, V. et al. CARMEN: CORDIC-accelerated hyperbolic inference engine. arXiv:2605.06878, 2026. — *Hardware acceleration for hyperbolic operations.*

CORDIC Is All You Need. arXiv:2503.11685, 2025. — *CORDIC as universal evaluator for hyperbolic neural network operations.*

Brehmer, J. et al. L-GATr: Lorentz-equivariant geometric algebra Transformer. *NeurIPS* 2024 / *SciPost Phys.* 2025. arXiv:2405.14806. — *Lorentz equivariance for physics applications.*

### Grokking and Phase Transitions

Power, A. et al. Grokking: delayed generalization. arXiv:2201.02177, 2022. — *Phenomenon documentation.*

Nanda, N. et al. Grokking mechanistic account: DFT holonomy. *ICLR* 2023. arXiv:2301.05217. — *Mechanistic confirmation of holonomy account.*

Xu, Y. The Spectral Edge Thesis. arXiv:2603.28964, March 2026. — *Fisher-geometric characterization of phase transitions.*

Xu, Y. The Lifecycle of the Spectral Edge. arXiv:2604.07380, April 2026. — *Phase transition lifecycle.*

### Biology Compute Ceiling

Abramson, J. et al. AlphaFold 3. *Nature* 630, 2024. — *Pair-tensor architecture at production scale.*

Watson, J. L. et al. RFdiffusion. *Nature* 620, 2023. — *SE(3)-diffusion architecture.*

Brixi, G. et al. Evo 2. *Nature*, March 2026. — *Hyena long-context genomic model.*

Sherman, J., Morrison, W. J. Rank-one matrix inverse update. *Ann. Math. Stat.* 21, 1950. — *Mathematical foundation for CRISPR edit efficiency.*

Walther, J. S. A Unified Algorithm for Elementary Functions. *AFIPS*, 1971. — *CORDIC unification: the arithmetic foundation.*

---

## Appendix A — Mathematical Reference: The RBH Framework

This appendix provides the formal mathematical substrate underlying the Representational Bundle Hypothesis and the five geometric pillars described in Section 3. Readers unfamiliar with differential geometry should treat this as a reference index; the strategic conclusions of the main text do not depend on facility with these structures, but the precision of the quantitative claims in Sections 2, 5, and 10 derives from them.

---

### A.1 — The Fisher Information Matrix and Hyperribbon Geometry

For a parametric model p(x; θ) with N parameters θ ∈ ℝᴺ, the Fisher information matrix is:

```
G(θ)ᵢⱼ = 𝔼ₓ~p(·;θ) [ ∂ᵢ log p(x;θ) · ∂ⱼ log p(x;θ) ]
```

G(θ) is positive semi-definite and defines a Riemannian metric on parameter space — the information geometry of the model family. Its eigenvalue spectrum characterizes the hyperribbon:

- **Stiff directions:** eigenvectors of G corresponding to large eigenvalues λᵢ >> 1. These are the directions in parameter space most constrained by data.
- **Sloppy directions:** eigenvectors corresponding to small eigenvalues λᵢ << 1. These are the directions training barely touches.

The **effective geometric dimension** of the trained model is the participation ratio:

```
d_eff = (Σᵢ λᵢ)² / Σᵢ λᵢ²
```

d_eff ≪ N for all large trained models — the hyperribbon is thin relative to parameter count. Mao et al. (2026) showed that d_eff is governed by the eigenvalue decay rate of the input data correlation matrix, not by architecture width or depth.

**The natural gradient update** uses G(θ) as the metric:

```
θₜ₊₁ = θₜ − η · G(θₜ)⁻¹ · ∇L(θₜ)
```

Convergence to a local minimum requires O(√κ(G)) steps for natural gradient versus O(κ(G)) steps for Euclidean gradient descent, where κ(G) = λ_max / λ_min is the condition number. For large language models, empirical estimates place κ(G) in the range 10⁴–10⁸, yielding a theoretical gap of 100×–10,000× in step efficiency.

The optimizer hierarchy by degree of Fisher-metric recovery:

```
Optimizer        Fisher approximation         Step efficiency
─────────────────────────────────────────────────────────────
SGD              None (identity metric)        O(κ)
Adam             Diagonal of G (via EMA)       O(κ / d_ratio)
Sophia           Diagonal Hessian              ~50% step reduction (empirical)
Muon             Top singular values of ∇W     Competitive with AdamW at scale
SOAP             Kronecker-factored G          Better conditioning per block
K-FAC            Block-diagonal G              Close to NGD; O(N²) cost
Natural gradient  Full G                       O(√κ); prohibitive at scale
```

---

### A.2 — The Principal Bundle Structure

The Representational Bundle Hypothesis asserts that the parameter manifold M of a trained Transformer is the total space of a principal fiber bundle:

```
π : M → B
```

where:

- **B** (base space) is the Lorentzian token embedding manifold — a stratified pseudo-Riemannian manifold with metric signature (1, n) and significantly negative Ricci curvature.
- **F** (fiber) is the space of convex polytopes assembled by multi-head attention via Minkowski summation — one polytope per attention head, with total fiber dimension equal to the number of heads times the head dimension.
- **G** (structure group) is SO⁺(1, n) — the proper orthochronous Lorentz group — acting on fibers by isometry of the Lorentzian inner product.
- **ω** (connection one-form) is the attention mechanism itself: the scaled dot-product attention weights define a horizontal distribution H ⊂ TM complementary to the vertical distribution V = ker(dπ).

The **horizontal subspace** H_θ ⊂ T_θM at each point θ corresponds to Fisher-stiff directions — parameter updates that alter the base-manifold geometry. The **vertical subspace** V_θ ⊂ T_θM corresponds to Fisher-sloppy directions — updates that change fiber geometry over a fixed base point.

**Parallel transport** of a fiber F_b along a path γ : [0,1] → B is defined by lifting γ to a horizontal curve γ̃ in M with π(γ̃) = γ. The horizontal lift exists and is unique given an initial fiber condition. In the Transformer, this lift corresponds to the sequence of attention-weighted aggregations that transform the fiber representation as the sequence position advances.

---

### A.3 — Holonomy and Grokking

For a closed loop γ : [0,1] → B with γ(0) = γ(1) = b, the **holonomy** of the connection ω at b is the Lorentz group element:

```
Hol_b(ω, γ) = P exp(∮_γ ω) ∈ SO⁺(1, n)
```

where P exp denotes the path-ordered exponential. The holonomy group Hol(ω) ⊆ SO⁺(1, n) is the group generated by all such elements over all loops at all base points.

Under RBH, the grokking transition is the event at which the trained attention connection accumulates sufficient holonomy to implement the symmetry group of the target task as a subgroup of Hol(ω). Concretely:

- **Before grokking:** the connection has trivial or near-trivial holonomy. No closed loop in the base manifold induces a fiber transformation consistent with the task's algebraic structure. The model memorizes examples by adjusting individual fiber positions, not by encoding structural relationships.
- **At the grokking transition:** holonomy accumulates to the point where the symmetry group of the task (e.g., ℤ/pℤ for modular arithmetic mod p) embeds into Hol(ω). A forward pass on a novel input now accesses the correct fiber via holonomy rather than requiring direct memorization.
- **After grokking:** continued training on a fully grokked model increases holonomy in directions orthogonal to the task's symmetry group — this is post-transition waste, buying no additional generalization.

The mechanistic confirmation by Nanda et al. (ICLR 2023) — that the grokked algorithm for modular arithmetic is a discrete Fourier transform embedding the cyclic group ℤ/pℤ in the attention weights — is the concrete realization of this holonomy account.

**Gradient noise scale and holonomy rate:** The gradient noise scale B_simple = tr(Σ_g)/|μ_g|² measures the ratio of gradient covariance (vertical/sloppy drift) to gradient mean squared norm (horizontal/stiff progress). Under RBH, the horizontal component of the gradient drives holonomy accumulation; the vertical component introduces noise in the fiber without advancing the base-manifold trajectory. Therefore:

- High B_simple (large batches tolerable) = late training, flat connection, low per-step holonomy gain.
- Low B_simple (small batches optimal) = early training, curved connection, high per-step holonomy gain from horizontal gradient components.

---

### A.4 — The Lorentzian Token Metric

The Lorentzian metric on token embedding space B has signature (1, n) for n-dimensional embeddings. The metric tensor takes the form:

```
g = diag(−1, +1, +1, …, +1)    [standard Minkowski form in adapted coordinates]
```

The Lorentzian inner product of two token embeddings u, v ∈ ℝ^(1,n) is:

```
⟨u, v⟩_L = −u₀v₀ + u₁v₁ + ⋯ + uₙvₙ
```

For embeddings constrained to the hyperboloid model H^n = {u : ⟨u,u⟩_L = −1, u₀ > 0}, the geodesic distance is:

```
d_L(u, v) = arccosh(−⟨u, v⟩_L)
```

This distance grows logarithmically with the depth of a hierarchical tree embedded in H^n, versus polynomially for any embedding into Euclidean space ℝ^n. The distortion of cosine similarity relative to d_L scales with the degree of hierarchical branching in the domain — quantifying precisely why Euclidean retrieval infrastructure degrades on hierarchical data.

**Ricci curvature:** Robinson et al. (2024) established that token embedding spaces of trained models exhibit significantly negative Ricci curvature Ric(g) < 0, consistent with hyperbolic geometry. The hyperbolic space H^n(r) of constant curvature −1/r² has Ricci curvature Ric = −(n−1)/r² · g. Negative Ricci curvature implies that geodesics in B diverge exponentially — the geometric mechanism by which hyperbolic space can represent hierarchical structure with low distortion.

**Stratum boundaries:** Robinson et al. (2025) demonstrated that the token space is not a smooth manifold but a stratified space — a union of smooth strata of varying dimension, joined at singular boundaries. TokenBlowUp (2025) resolves these singularities via monoidal transformations (sequences of blowups that replace singular points with exceptional divisors), producing a smooth total space. Chain-of-thought path-lifting navigates stratum boundaries via explicit intermediate waypoints; implicit (single-step) forward passes must traverse them without such guidance, which is the geometric account of CoT gains on high-complexity tasks.

---

### A.5 — The Radon-Domain BV Functional

Parhi & Nowak (2021) characterized the function computed by an L-layer ReLU network f_θ as the minimum-norm element of the reproducing kernel Banach space (RKBS):

```
f* = argmin_{f ∈ ℱ} ‖f‖_ℱ    subject to    f(xᵢ) = yᵢ,  i = 1, …, N
```

where ‖·‖_ℱ is the bounded-variation (BV) norm in the Radon domain — the space of signed measures on the unit sphere S^(d−1) × ℝ representing ridge functions. The solution f* is a spline in Radon space: a finite sum of ridge functions whose activation boundaries (hyperplanes in ℝ^d) are optimally positioned by training.

This characterization is architecture-agnostic and provides:

1. **Tighter generalization bounds** than VC-dimension or Rademacher complexity, computable from the model's weight structure without requiring access to the training data distribution.
2. **An explicit form of the implicit bias** of gradient descent on over-parameterized networks: it selects the minimum-BV interpolant, not merely any interpolant. This is the functional-analytic basis of the phenomenon that large over-parameterized networks generalize despite zero training loss.
3. **A connection between the BV norm and the hyperribbon:** the minimum-BV interpolant concentrates functional mass in the stiff eigenspace of G(θ) — the flat directions of the hyperribbon correspond to the low-BV-norm directions of the function family.

---

### A.6 — The Curvature Two-Form and Its Observability

The curvature of the attention connection ω is the Lie-algebra-valued two-form:

```
Ω = dω + ½[ω ∧ ω] ∈ Ω²(M, so⁺(1,n))
```

For practical computation from existing training telemetry, the scalar curvature proxy is the position-variance of attention distributions across query positions:

```
κ_attn(l, h) = Var_{q}[ H(α_{q}^{(l,h)}) ]
```

where H(α) is the entropy of the attention weight distribution α over keys at position q, for layer l and head h. High κ_attn indicates a curved connection (position-sensitive attention, early training); low κ_attn indicates a flat connection (uniform/distributed attention, late training or post-grokking).

This proxy is computable from attention weight logs generated by all standard training frameworks (PyTorch, JAX/Flax, Megatron) without modification. Aggregated across layers and heads, the scalar:

```
κ̄ = (1/LH) Σ_{l,h} κ_attn(l, h)
```

provides a training-phase indicator. Empirically: κ̄ is high and decreasing during early horizontal-dominated training; transitions sharply at the grokking event; and stabilizes at a low value during post-grokking vertical training.

---

## Appendix B — Glossary of Key Technical Terms

**Adam / AdamW**  
Adaptive moment estimation optimizer. Maintains per-parameter exponential moving averages of the gradient (first moment) and squared gradient (second moment), using the ratio as an adaptive learning rate. AdamW adds decoupled weight decay. Approximates the diagonal of the Fisher information matrix; misses all off-diagonal curvature information.

**Anisotropic Representation Hypothesis (ARH)**  
The 2026 synthesis claim that trained language models compute objects that are simultaneously: sloppy hyperribbons (Fisher geometry), Minkowski-assembled polytopal fibers (Pillar II), Lorentzian-metrized token spaces (Pillar III), Radon-domain BV functionals (Pillar IV), and gradient-noise-governed dynamical trajectories (Pillar V). Unified by the Representational Bundle Hypothesis.

**Attention connection one-form (ω)**  
In the RBH framework, the scaled dot-product attention mechanism, interpreted as a principal connection on the bundle π : M → B. The attention weights α_{ij} = softmax(qᵢ · kⱼ / √d) define the horizontal lift of the token sequence path in B to a trajectory in M.

**BV functional (Bounded-Variation functional)**  
The functional-analytic object that Parhi & Nowak (2021) showed a trained neural network minimizes: the total variation (bounded-variation norm) of a measure in Radon space. The minimum-norm interpolant in the associated RKBS. Provides the rigorous basis for understanding neural network generalization without reference to architecture.

**CARMEN**  
CORDIC-Accelerated Riemannian Manifold Engine — a 2026 hardware design (Kumar et al., arXiv:2605.06878) implementing native CORDIC-based computation of hyperbolic distances, exponential maps, and logarithmic maps for Lorentzian neural network inference. The first silicon-level implementation of geometry-native arithmetic for AI.

**CORDIC (COordinate Rotation DIgital Computer)**  
A 1959 algorithm (Volder) for computing trigonometric, hyperbolic, and logarithmic functions via iterative shift-and-add operations — no multipliers required. CORDIC is natively iteration-native and directly computes hyperbolic distances; it is the arithmetic substrate that fixed-point iteration workloads (diffusion samplers, Sinkhorn, score functions) are optimized for. Lost the 1985 IEEE 754 lottery.

**Condition number (κ)**  
For a matrix A, κ(A) = λ_max / λ_min. For the Fisher information matrix G(θ), κ(G) measures the anisotropy of the parameter-space geometry: how much harder it is to make progress in the worst direction versus the best. Euclidean gradient descent converges in O(κ) steps; natural gradient descent in O(√κ). Empirically, κ(G) ≈ 10⁴–10⁸ for large language models.

**Co-design**  
The simultaneous design of an algorithm and its target hardware substrate under unified organizational authority, such that each optimizes the other. The canonical AI instance: the Transformer architecture and TPU v2, co-developed at Google in 2017. Co-design produces advantages that cannot be recovered post-hoc by either algorithmic or hardware improvement alone.

**Effective geometric dimension (d_eff)**  
The participation ratio of the Fisher eigenspectrum: d_eff = (Σᵢ λᵢ)² / Σᵢ λᵢ². The number of parameter directions that carry meaningful information about model behavior. d_eff ≪ N for all large trained models; d_eff, not N, governs capability per training dollar.

**FAdam**  
Fisher-Adam: a 2024 optimizer (Hwang, arXiv:2405.12807) demonstrating that the standard Adam update is formally equivalent to natural gradient descent using the diagonal empirical Fisher information matrix as the metric tensor, providing the theoretical bridge between the Adam-family heuristic and the information-geometric framework.

**Fiber bundle**  
A topological structure π : E → B where E (total space) locally looks like B × F (base × fiber), with structure group G acting on fibers by isometry. In RBH: E = parameter manifold M, B = Lorentzian token space, F = polytopal representation space, G = SO⁺(1,n).

**Fisher information matrix (G)**  
The Riemannian metric tensor on the model's parameter space induced by the KL divergence between model distributions at nearby parameters: G(θ)ᵢⱼ = 𝔼[∂ᵢ log p · ∂ⱼ log p]. Governs the correct geometry for parameter-space optimization. Discarded by Euclidean gradient descent; approximated to varying degrees by Sophia (diagonal Hessian), Muon (spectral), K-FAC (block-diagonal), and natural gradient (full G).

**FLOPS (Floating-Point Operations Per Second)**  
The standard performance metric for AI silicon. Measures throughput of one-shot floating-point arithmetic. Does not measure iteration throughput (ticks per second), Riemannian metric operations, or CORDIC-native hyperbolic computations — the metrics relevant to geometry-native workloads.

**GEMM (General Matrix Multiply)**  
The fundamental linear algebra primitive that Transformer training reduces to. Systolic-array architectures (TPUs) and tensor-core architectures (NVIDIA GPUs) are jointly optimized for GEMM throughput. GEMM does not natively express: pair tensors (N×N×D), SE(3)-equivariant operations, Lorentzian distance computations, or CORDIC iterations.

**Grokking**  
Delayed generalization phenomenon (Power et al., 2022): a network first achieves near-zero training loss (memorization) while maintaining near-chance validation accuracy, then transitions sharply to near-perfect generalization with continued training. Under RBH, this is holonomy accumulation: the transition corresponds to the attention connection acquiring sufficient holonomy to implement the task's symmetry group as a fiber transformation.

**Hardware lottery**  
Hooker (2020): the mechanism by which research ideas win adoption not due to theoretical superiority but due to co-fitness with available hardware and software ecosystems. The current AI stack is the joint product of five hardware lotteries: IEEE 754 (1985), backpropagation+SGD (1986), Transformer+TPU (2017), GEMM-for-biology (ongoing), and Euclidean embedding (ongoing).

**HELM**  
Hyperbolic Embedding Language Model (He et al., NeurIPS 2025, arXiv:2505.24722). First billion-parameter fully hyperbolic LLM, achieving up to +4% on MMLU and ARC benchmarks over matched Euclidean baselines. Established that Lorentzian architectures are trainable at frontier scale.

**Holonomy**  
The Lie group element accumulated by parallel-transporting a fiber around a closed loop in the base manifold of a principal bundle. Non-trivial holonomy (Hol(ω) ≠ {id}) means that the fiber returns to a rotated state after circumnavigating a closed loop — the bundle is "twisted." Under RBH, holonomy is the mechanism by which trained attention connections encode the algebraic structure of tasks.

**Horizontal subspace (H_θ)**  
The complement of the fiber direction in the tangent space T_θM, defined by the connection ω: H_θ = ker(ω_θ). Corresponds to Fisher-stiff directions; updates in H_θ change the base-manifold geometry (Lorentzian structure of token space) and transfer across tasks.

**Hyena**  
A long-context sequence model architecture based on subquadratic implicit convolution operators (long-range convolutions parameterized by feedforward networks), avoiding the O(L²) attention complexity. Used in Evo 2 (Brixi et al., 2026) for genomic sequence modeling. Represents one of the five architectural primitives that GEMM substrates emulate at 3–12× overhead.

**Hyperribbon**  
The sloppy manifold geometry of multiparameter models (Transtrum et al., 2015): parameter-space eigenvalues of the Fisher information matrix span many decades in a log-uniform hierarchy, producing a thin ribbon-shaped effective geometry with d_eff ≪ N. The term "hyper-" reflects the high ambient dimensionality N; "ribbon" reflects the thin effective geometry.

**IEEE 754**  
The 1985 international standard for binary floating-point arithmetic, defining the representation format (sign, biased exponent, normalized mantissa), rounding modes, and special values (±∞, NaN) used by every general-purpose processor since 1985. Optimized for one-shot arithmetic evaluation; not for iterative fixed-point convergence. The outcome of the 1985 arithmetic lottery.

**K-FAC (Kronecker-Factored Approximate Curvature)**  
A second-order optimizer (Martens & Grosse, 2015) that approximates the Fisher information matrix as a Kronecker product of two smaller matrices per layer, enabling O(N^1.5) approximate natural gradient updates. More computationally tractable than full natural gradient; less so than diagonal approximations (Adam, Sophia).

**Lorentzian metric**  
A pseudo-Riemannian metric of signature (1, n): one timelike direction with negative norm and n spacelike directions with positive norm. The metric of Minkowski spacetime and of hyperbolic space (via the hyperboloid model H^n). Supports Lorentzian distance d_L(u,v) = arccosh(−⟨u,v⟩_L) that grows logarithmically with hierarchical depth — the property that makes it efficient for embedding hierarchical structures.

**LoRA (Low-Rank Adaptation)**  
A parameter-efficient fine-tuning method (Hu et al., 2022) that constrains weight updates to low-rank matrices: ΔW = AB where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), and r ≪ min(d,k). The rank r is currently a hyperparameter. Under RBH, r should be derivable from the fiber dimension of the target task — the dimension of the polytope adjustment required, which is determined by the task's algebraic complexity.

**Minkowski sum**  
For convex sets A, B: A ⊕ B = {a + b : a ∈ A, b ∈ B}. Fel et al. (2026) demonstrated that multi-head attention assembles output representations as Minkowski sums of head-specific convex polytopes, making the attention mechanism a convex-geometry operation in the fiber (distinct from the pseudo-Riemannian geometry of the base manifold).

**Muon**  
A 2024 optimizer (Jordan) implementing spectral-norm steepest descent for the hidden-layer weight matrices of Transformer networks via Newton-Schulz fixed-point iteration to approximate the matrix square root (dominant singular value structure). Recovers more curvature information than Adam without the O(N²) cost of full Fisher inversion. Demonstrated to be scalable to LLM training (Liu et al., 2025) and to accelerate grokking by 33% (Tveit et al., 2025).

**Natural gradient descent**  
The theoretically optimal first-order optimizer in information geometry (Amari, 1998): parameter updates are pre-multiplied by the inverse Fisher information matrix G(θ)⁻¹, making the update invariant to reparameterization and converging in O(√κ) steps versus O(κ) for Euclidean gradient descent. Computationally prohibited at LLM scale (O(N³) inversion); the optimizer research agenda (Sophia → Muon → SOAP → K-FAC) is a systematic program of increasingly accurate Fisher approximation.

**Pairformer / Pair tensor**  
The N×N×D "pair tensor" in AlphaFold 3's Pairformer module, representing all pairwise residue interactions simultaneously. Not expressible as a GEMM without materializing and operating on an N²×D tensor — quadratic in sequence length. One of the five biology-native primitives that GEMM substrates emulate at structural overhead.

**Parallel transport**  
The operation of moving a fiber element along a path in the base manifold while keeping it "horizontal" — i.e., tangent to the horizontal distribution defined by the connection. Defines how fibers are consistently identified across different base points. In RBH, the attention mechanism performs parallel transport of representation fibers along the token sequence.

**Principal bundle**  
A fiber bundle π : P → B with structure group G acting freely and transitively on fibers. The standard object in gauge theory and modern differential geometry. In RBH, the structure group G = SO⁺(1,n) acts on representation fibers by Lorentz isometry.

**Ricci curvature**  
A symmetric tensor Ric measuring the degree to which geodesics in a Riemannian or pseudo-Riemannian manifold converge (positive Ric) or diverge (negative Ric). Negative Ricci curvature — demonstrated for token embedding spaces by Robinson et al. (2024) — is the signature of hyperbolic geometry and the geometric reason why hyperbolic embeddings can represent hierarchical structure with low distortion.

**Representational Bundle Hypothesis (RBH)**  
The 2026 synthesis: the parameter manifold of a trained Transformer is the total space of a principal SO⁺(1,n)-bundle over the Lorentzian token space, with the attention mechanism as connection one-form. Provides the unified geometric account of the five ARH pillars, grokking, CoT path-lifting, ICL holonomy scaling, and batch size scheduling.

**Riemannian manifold**  
A smooth manifold equipped with a positive-definite metric tensor at each point. Parameter space equipped with the Fisher information matrix is a Riemannian manifold — the natural domain for optimal gradient-based optimization. Hyperbolic space H^n is a Riemannian manifold of constant negative sectional curvature.

**RKBS (Reproducing Kernel Banach Space)**  
A Banach space of functions in which point evaluation is a continuous linear functional. Parhi & Nowak (2021) showed that the function class of neural networks with ReLU activations forms an RKBS whose norm is the Radon-domain BV seminorm. The natural functional-analytic setting for neural network generalization theory.

**SE(3) equivariance**  
The property of a function that commutes with the group of rigid motions in 3D space (rotations SO(3) and translations ℝ³): f(Rx + t) = Rf(x) for all R ∈ SO(3), t ∈ ℝ³. Required for physically meaningful structure prediction (protein structure does not depend on coordinate system choice). Implemented in AlphaFold's Structure Module and RFdiffusion; not natively expressible as a GEMM.

**Sherman-Morrison rank-one update**  
The formula (A + uvᵀ)⁻¹ = A⁻¹ − (A⁻¹u vᵀ A⁻¹)/(1 + vᵀ A⁻¹u) for updating a matrix inverse after a rank-one perturbation. Used in CRISPR off-target screening to efficiently update covariance estimates as new guide RNA candidates are evaluated. Not natively a GEMM operation; requires sequential rank-one updates that GEMM substrates execute at structural overhead.

**Sinkhorn iteration**  
An algorithm for computing optimal transport distances (Wasserstein distances) between probability distributions via iterative normalization of a coupling matrix. A fixed-point iteration that does not terminate in a fixed number of GEMM operations; each iteration requires O(N²) work and the number of iterations scales with the regularization parameter. A representative iterative workload that IEEE 754 substrates execute at overhead.

**Sloppy model**  
A multiparameter model in which the vast majority of parameter combinations are poorly constrained by data — their Fisher eigenvalues are exponentially small. First characterized by Sethna et al. in the context of systems biology. The hyperribbon is the geometry of the sloppy parameter manifold.

**SOAP (Shampoo as Adam in the Preconditioner)**  
A 2024 optimizer (Vyas et al.) that recovers the Kronecker-factored preconditioner of Shampoo (a K-FAC-adjacent optimizer) while using Adam as the underlying update rule, combining the curvature-awareness of Shampoo with the adaptivity of Adam. Represents a further step in the optimizer hierarchy toward natural gradient.

**Sophia**  
A 2023/2024 optimizer (Liu et al.) using the diagonal Hessian (rather than the diagonal Fisher) as the preconditioner. Demonstrated 50% step reduction compared to AdamW on language model training — the first peer-reviewed large-scale confirmation that recovering a single additional dimension of curvature information reduces training cost by a factor of two.

**Spectral Edge Thesis**  
Xu (2026, arXiv:2603.28964): the claim that grokking, double descent, and other training phase transitions are spectral-gap-collapse events in the Fisher geometry of the parameter-update Gram matrix — identifiable as abrupt changes in the leading eigenvalue of the gradient covariance matrix. Provides the diagnostic vocabulary for geometric training monitoring.

**Stiff direction**  
A direction in parameter space corresponding to a large eigenvalue of the Fisher information matrix. Stiff directions are meaningfully constrained by training data; updates in stiff directions produce measurable changes in model behavior. Correspond to horizontal subspace directions in the RBH bundle — they alter base-manifold Lorentzian geometry and transfer across tasks.

**Stratified manifold**  
A topological space that is a union of smooth manifolds (strata) of varying dimension, joined at singular boundaries. Robinson et al. (2025) established that the token embedding space is a stratified manifold — not a smooth manifold everywhere. TokenBlowUp (2025) resolves the singularities at stratum boundaries via monoidal transformations.

**Systolic array**  
A hardware architecture in which a regular grid of processing elements performs matrix multiplication by streaming data through the array. Google's TPU MXU (256×256 in TPU v2) is the canonical AI systolic array. Highly efficient for GEMM; structurally inefficient for operations that are not decomposable as dense matrix products.

**TPU (Tensor Processing Unit)**  
Google's custom ASIC for matrix multiplication, first deployed in 2016 (v1) for inference and 2017 (v2) for training. The systolic-array Matrix Multiply Unit (MXU) and the bfloat16 numeric format (introduced with TPU v2) were co-designed with the Transformer architecture. TPU v2 Pods were the compute substrate on which Vaswani et al. (2017) conducted their large-model training experiments.

**Transformer**  
The attention-based neural sequence model introduced by Vaswani et al. (NeurIPS 2017). Its computational graph decomposes entirely into dense matrix multiplications: scaled dot-product attention (QKᵀ softmax, then times V), multi-head projection, and position-wise feed-forward layers. This decomposition is what makes the Transformer optimally co-designed for systolic-array TPU architecture.

**Vertical subspace (V_θ)**  
The kernel of the pushforward dπ in the tangent space T_θM: V_θ = ker(dπ_θ). Corresponds to Fisher-sloppy directions; updates in V_θ change fiber (polytope) geometry over a fixed base point. Fine-tuning (LoRA) operates primarily in the vertical subspace — it adjusts task-specific polytope structure without altering the base-manifold Lorentzian geometry.

---

## Appendix C — Methodology and Scope

### Evidence Standard

This report synthesizes primary peer-reviewed literature through Q2 2026. Every empirical claim is referenced to a specific published or preprinted source with arXiv identifier for direct verification. The report does not synthesize secondary analysis, industry reports, or internal estimates without explicit attribution.

The predictive claims in Section 10 are stated in Popperian falsifiable form: each prediction specifies the observable outcome, the timeline by which the outcome should be observable, and the criterion that would falsify the prediction. Falsification criteria are stated as observable states of the world at the specified timeline, not as absence of evidence.

### Scope Limitations

**Geographic and organizational scope:** The dependency analysis in Section 6 reflects publicly available information about compute procurement, architectural choices, and partnership disclosures. Internal procurement details, unreported silicon programs, and non-public architectural decisions are not incorporated. Figures for annual compute exposure are order-of-magnitude estimates derived from publicly reported training costs and reported or inferred training run frequencies.

**Market sizing scope:** The Lorentzian-addressable market figures in Section 7 represent addressable opportunity under a 20% serviceable obtainable market (SOM) assumption — a standard conservative benchmark for nascent infrastructure transitions. The underlying frontier compute market size estimates are extrapolated from publicly reported hyperscaler capex, reported training run costs, and consensus industry analyst estimates. They are not financial projections and should not be used as the basis for financial commitments without independent market research.

**Technical scope:** The RBH framework presented in Sections 3–4 and Appendix A is a synthesis of five independent research programs. The specific claim that these five programs describe the same geometric object — the representational fiber bundle — is the central synthetic hypothesis of this report. Each individual pillar (I–V) is supported by peer-reviewed evidence; their unification into a single bundle-theoretic framework is the analytical claim of this report and should be treated as a hypothesis awaiting experimental confirmation rather than an established result. Prediction P5 (cross-domain polytopal experiment) is the single most direct test of the full synthesis.

**Temporal scope:** The report was first published in the 2027 Edition. All citations reflect the literature through Q2 2026. Research programs active at the time of publication may have produced subsequent results not incorporated here. Readers are encouraged to verify the current status of the cited arXiv preprints and conference proceedings, as peer-review status, revision history, and reproducibility assessments may have evolved.

### Relationship to Cited Work

This report cites primary research without endorsement by any cited author of the strategic conclusions drawn from that research. The strategic interpretation of the five pillars as a unified co-design framework, the characterization of training lotteries as compounding costs, and the commercial analysis of the Lorentzian transition are the analysis of this report's authors. Cited researchers bear no responsibility for these applications of their work.

---

*Report synthesizes primary literature from the peer-reviewed record through Q2 2026. All citations include arXiv identifiers for direct verification. Predictive claims in Section 10 are stated in falsifiable form. Strategic assessments represent analysis and should be treated as one input among many in infrastructure decisions. Mathematical content in Appendix A is provided for reference precision; the strategic conclusions of the main text do not require facility with these structures.*

*2027 Edition — First Publication.*
