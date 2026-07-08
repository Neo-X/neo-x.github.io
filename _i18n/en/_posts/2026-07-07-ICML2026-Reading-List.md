---
title: "My ICML 2026 Reading List: RL, Robot Learning, VLA & World Models"
date: 2026-07-07
description: "A prioritized reading list of ~90 ICML 2026 papers on reinforcement learning, robot learning, imitation learning, VLA models, diffusion for RL, world models, and generalization — filtered from 6,627 accepted papers."
summary: "I filtered all 6,627 ICML 2026 papers down to a prioritized list for my own research areas — RL, robot learning, VLA models, imitation learning, diffusion for RL, world models, and generalization — plus a handful of relevant position papers. Sharing it here in case it's useful to others heading to Seoul."
category: Article
tags:
   - reinforcement-learning
   - robot-learning
   - imitation-learning
   - vla
   - world-models
   - icml
author: Glen Berseth
authors: Glen Berseth
draft: false
layout: page
type: Reading List
titleShort: ICML 2026 Reading List
---

# My ICML 2026 Reading List

ICML 2026 accepted 6,627 papers. Nobody is reading all of them, so I filtered the list down to what's relevant to my group's research: **reinforcement learning, robot learning, imitation learning, vision-language-action (VLA) models, diffusion for RL, world models, and generalization.** I'm sharing the filtered list here in case it's useful to others heading to Seoul.

Papers are grouped into **P0 (must read)**, **P1 (should read)**, and **P2 (nice to read)**. Where a paper already has an arXiv preprint (✅) I've linked it directly; otherwise use the OpenReview link for the PDF.

<!--more-->

## P0 — Must Read

### VLA Models
- NeurVLA: Unleashing Failure-Handling Capability of VLMs via Neural-Symbolic Reasoning — [OpenReview](https://openreview.net/forum?id=V0OaAHqBUy)
- StableVLA: Towards Robust VLMs without Extra Data — [OpenReview](https://openreview.net/forum?id=aOSkHBRYUJ)
- XR-1: Versatile VLMs via Unified Vision-Motion Representations — [OpenReview](https://openreview.net/forum?id=JO0IsGJg16)
- Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds — [OpenReview](https://openreview.net/forum?id=zyMvoKYWMZ)
- LARA: Latent Action Representation Alignment for VLMs — [OpenReview](https://openreview.net/forum?id=sQTzABTEQM)
- From Pixels to Tokens: Systematic Study of Latent Action Supervision for VLMs — [OpenReview](https://openreview.net/forum?id=VMsumctGvg)
- Can VLMs Diagnose and Recover from VLA Manipulation Faults? — [OpenReview](https://openreview.net/forum?id=Q0rAQZg9t5)

### Robot Learning & Imitation
- ✅ Posterior Behavioral Cloning: Pretraining BC for Efficient RL Fine-tuning — [arXiv:2512.16911](https://arxiv.org/abs/2512.16911)
- ✅ NavOL: Navigation Policy with Online Imitation Learning — [arXiv:2605.11762](https://arxiv.org/abs/2605.11762)
- ✅ Noise-Guided Transport: Imitation Learning from Random Priors — [arXiv:2509.26294](https://arxiv.org/abs/2509.26294)
- Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning — [OpenReview](https://openreview.net/forum?id=BJUEgspMUt)
- Towards Practical World Model-based RL for VLA Models — [OpenReview](https://openreview.net/forum?id=yKQ8GrwEhr)
- ✅ Temporal Difference Calibration for VLA Models — [arXiv:2604.20472](https://arxiv.org/abs/2604.20472)
- Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL — [OpenReview](https://openreview.net/forum?id=qgAKuqzYBC)

### Diffusion for RL
- Energy-based Compositional Diffusion Planning — [OpenReview](https://openreview.net/forum?id=r2sJKlXY3M)
- ✅ Improving Diffusion Planners by Self-Supervised Action Gating with Energies — [arXiv:2603.02650](https://arxiv.org/abs/2603.02650)
- Latent Diffusion Controller: Framework, Algorithms and Parameterization — [OpenReview](https://openreview.net/forum?id=IGTMjtehxq)
- ✅ Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge — [arXiv:2512.07212](https://arxiv.org/abs/2512.07212)
- ✅ SVL: Goal-Conditioned RL as Survival Learning — [arXiv:2604.17551](https://arxiv.org/abs/2604.17551)

### RL Theory & Policy Optimization
- ✅ Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL — [arXiv:2602.03389](https://arxiv.org/abs/2602.03389)
- ✅ Why Linear Recurrent Memory Works in Partially Observable RL — [arXiv:2605.31261](https://arxiv.org/abs/2605.31261)
- Towards Optimal Strong Regret and Constraint Violation via Model-free RL — [OpenReview](https://openreview.net/forum?id=y8rIspGlNQ)
- ✅ Learning to Perceive the World Through Control: Empowerment-Based Representation — [arXiv:2605.30656](https://arxiv.org/abs/2605.30656)
- ✅ Reparameterization Flow Policy Optimization — [arXiv:2602.03501](https://arxiv.org/abs/2602.03501)
- Reparameterization PPO — [OpenReview](https://openreview.net/forum?id=TknqRTyQ0a)

## P1 — Should Read

### World Models
- Structured 4D Latent World Model for Robot Planning — [OpenReview](https://openreview.net/forum?id=aXAgpGfHGc)
- Learning Latent Action World Models In The Wild — [OpenReview](https://openreview.net/forum?id=HDf5semiaB)
- VectorWorld: Efficient Streaming World Model via Diffusion Flow — [OpenReview](https://openreview.net/forum?id=yTEEiE3YtD)
- Convergent World Representations and Divergent Tasks — [OpenReview](https://openreview.net/forum?id=Z4PjCncxzz)

### Offline RL
- ✅ BiTrajDiff: Bidirectional Trajectory Generation with Diffusion for Offline RL — [arXiv:2506.05762](https://arxiv.org/abs/2506.05762)
- VIPO: Value Function Inconsistency Penalized Offline RL — [OpenReview](https://openreview.net/forum?id=XAZdspeB5d)
- Fast Policy Learning for Offline RL via Bootstrapped Flow Q-Learning — [OpenReview](https://openreview.net/forum?id=2rGr38p5az)
- Offline RL with Generative Trajectory Policies — [OpenReview](https://openreview.net/forum?id=LpqtiYpHzn)
- Offline RL with Universal Horizon Models — [OpenReview](https://openreview.net/forum?id=GFmJ5XcGWl)
- Compositional Transduction for Offline Goal-Conditioned RL — [OpenReview](https://openreview.net/forum?id=OjXimqfwlX)
- Reward-Preserving Counterfactual State Editing for Offline RL — [OpenReview](https://openreview.net/forum?id=6D3IJpBGBW)
- SMAC: Score-Matched Actor-Critics for Offline-to-Online Transfer — [OpenReview](https://openreview.net/forum?id=OJSCaDIIJK)

### Diffusion RL
- Distillation Models are Good Samplers for Diffusion RL — [OpenReview](https://openreview.net/forum?id=VkMWujvv2c)
- Advantage Weighted Matching: Aligning RL with Diffusion Pretraining — [OpenReview](https://openreview.net/forum?id=nLY2pOYBrJ)
- Reverse Flow Matching: Unified Framework for Online RL with Diffusion Policies — [OpenReview](https://openreview.net/forum?id=vUpEe4yd1T)
- Trust-Region Diffusion Policies for Massively Parallel On-Policy RL — [OpenReview](https://openreview.net/forum?id=mGu2fs7kJt)

### Safe/Robust RL
- Distributionally Robust RL with Human Feedback — [OpenReview](https://openreview.net/forum?id=6GeYRoYKWP)
- CSPO: Constraint-Sensitive Policy Optimization for Safe RL — [OpenReview](https://openreview.net/forum?id=3ySR3TCMRP)
- Robust RL in a Sample-Efficient Setting (TMLR) — [poster](https://icml.cc/virtual/2026/poster/68794)
- Model-Free Robust Average-Reward RL with Sample Complexity — [OpenReview](https://openreview.net/forum?id=GMIHHrJ6Wp)
- Mirror Descent Policy Optimisation for Robust Constrained MDPs (TMLR) — [poster](https://icml.cc/virtual/2026/poster/68816)

### Continual/Meta/Transfer Learning
- SABER: Continual Learning with Representation Conflict Management — [OpenReview](https://openreview.net/forum?id=N0qnrJEIoy)
- Counterfactual Bootstrap for Robust Meta-RL — [OpenReview](https://openreview.net/forum?id=c4abksRFwY)
- Meta-learning Structure-Preserving Dynamics — [OpenReview](https://openreview.net/forum?id=k66TZFhUSQ)
- Dynamics Are Learned, Not Told: Zero-Shot Policy Adaptation — [OpenReview](https://openreview.net/forum?id=XQLa5PVQ0D)
- Motion Dynamics Learning for Few-Shot Embodied Adaptation — [OpenReview](https://openreview.net/forum?id=EW7FmahpLs)

### Multi-Agent/Hierarchical RL
- Hierarchical Policy Learning via Spectral Decomposition — [OpenReview](https://openreview.net/forum?id=hyw7WLPZae)
- Recurrent Structural Policy Gradient for Partially Observable Mean Field Games — [OpenReview](https://openreview.net/forum?id=VkZQThGNgI)
- Offline Multi-agent Continual Cooperation via Skill Partition and Reuse — [OpenReview](https://openreview.net/forum?id=5kteupXJ7B)
- Provably Convergent Actor-Critic in Risk-averse MARL — [OpenReview](https://openreview.net/forum?id=NpguYNGrG2)
- HyPOLE: Hyperproperty-Guided MARL under Partial Observation — [OpenReview](https://openreview.net/forum?id=EVoPYB6ss3)

### Additional
- Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RL — [OpenReview](https://openreview.net/forum?id=Xo95FS2GTK)
- Motion Attribution for Video Generation — [OpenReview](https://openreview.net/forum?id=zAl9heLw4q)
- Debate2Create: Robot Co-design via Multi-Agent LLM Debate — [OpenReview](https://openreview.net/forum?id=1ufVo73uzD)
- GAE: Unleashing Physical Potential of VLM with Generalizable Action Expert — [OpenReview](https://openreview.net/forum?id=6lq2MGo42H)
- EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance — [OpenReview](https://openreview.net/forum?id=luykjYIvEs)

## P2 — Nice to Read

### Flow Matching & Diffusion
- Flow for Future: SE(3)-Equivariant Flow Matching for 3D Trajectory Prediction — [OpenReview](https://openreview.net/forum?id=EBujA4tldV)
- Diffusion Bridge or Flow Matching? A Unifying Framework — [OpenReview](https://openreview.net/forum?id=aIFgQusnPy)
- Scaling the Prior: Size-Consistent Geometric Diffusion for 3D Molecules — [OpenReview](https://openreview.net/forum?id=5aH5eYIAFQ)
- Efficient, Property-Aligned Fan-Out Retrieval via RL-Amortized Diffusion — [OpenReview](https://openreview.net/forum?id=4P9cEcinYP)
- Well-Posed KL-Regularized Control via Wasserstein Divergences — [OpenReview](https://openreview.net/forum?id=diF53wYIj3)

### Representation / Efficient Fine-tuning
- What Does Preference Learning Recover from Pairwise Comparison Data? — [OpenReview](https://openreview.net/forum?id=WepBmd1L4v)
- Understanding LoRA as Knowledge Memory — [OpenReview](https://openreview.net/forum?id=0MVEJk4dhE)
- FedTreeLoRA: Federated LoRA Fine-Tuning — [OpenReview](https://openreview.net/forum?id=g3Hrh5aoal)
- Spectral Bridge VI: Dynamic LoRA via Bures-Wasserstein Gradient Flows — [OpenReview](https://openreview.net/forum?id=KjF35IhQGS)
- XTransfer: Modality-Agnostic Few-Shot Model Transfer — [OpenReview](https://openreview.net/forum?id=S1rq8s4FCA)
- Multi-Way Representation Alignment — [OpenReview](https://openreview.net/forum?id=goFOelWmEG)
- Revisiting Parameter-Based Knowledge Editing in LLMs — [OpenReview](https://openreview.net/forum?id=rxWzzqvHYZ)

## Position Papers

ICML 2026 had 65 position papers this year. Six stood out as relevant to RL / robot learning / world models:

**Must read:**
- *Position: World Models as an Intermediary between Agents and the Real World* — Sherry Yang
- *Position: RL Researchers Need to Distinguish Between Solving Simulators and Using Simulators as a Proxy* — Matthew Vandergrift, Esraa Elelimy, Martha White

**Should read:**
- *Position: Good Embodied Reward Models Need Bad Behavior Data* — Thomas Tian, Yilin Wu, Andrea Bajcsy
- *Position: Make Planning Research Rigorous Again!* — Michael Katz, Harsha Kokel, Christian Muise, Shirin Sohrabi, Sarath Sreedharan

**Nice to read:**
- *Position: Assistive AI requires Personalized Specialists, not Generalists* — Homanga Bharadhwaj
- *Position: Interestingness is an Inductive Heuristic for Future Compression Progress* — Vincent Herrmann, Jürgen Schmidhuber

(Search these titles on [OpenReview](https://openreview.net/group?id=ICML.cc/2026/Conference) — arXiv/OpenReview links for position papers were gated behind the ICML virtual-site login at the time of writing.)

## How I Built This List

I started from the full accepted-papers list (6,627 papers) and filtered by keyword/topic match against my group's research areas — RL, robot learning, VLA, imitation learning, diffusion for RL, generalization, and world models — then ranked by relevance into P0/P1/P2. arXiv preprints (✅) were matched where available; everything else links through OpenReview or the ICML virtual poster page. If you're at ICML 2026 in Seoul and this overlaps with your interests, hope it saves you some filtering time.
