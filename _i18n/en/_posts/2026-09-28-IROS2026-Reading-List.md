---
title: "My IROS 2026 Reading List: Robot Learning, VLAs, RL Fine-Tuning & World Models"
date: 2026-09-28
description: "A prioritized reading list of 171 IROS 2026 papers on robot learning, VLA models, RL fine-tuning of robot policies, imitation learning, diffusion and flow policies, world models, and generalization, filtered from 1,933 papers using their abstracts, plus the relevant workshops."
summary: "I filtered all 1,933 IROS 2026 papers down to a prioritized list for my own research areas — robot learning, VLA models, RL fine-tuning of robot policies, imitation learning, diffusion/flow policies, world models, and generalization — plus the related workshops. Sharing it here in case it's useful to others in Pittsburgh."
category: Article
tags:
   - reinforcement-learning
   - robot-learning
   - imitation-learning
   - vla
   - world-models
   - iros
author: Glen Berseth
authors: Glen Berseth
draft: false
layout: page
type: Reading List
titleShort: IROS 2026 Reading List
---

# My IROS 2026 Reading List

IROS 2026 in Pittsburgh has 1,933 papers in its program. Nobody is reading all of them, so I filtered the list down to what's relevant to my group's research: **robot learning, vision-language-action (VLA) models, RL fine-tuning of robot policies, imitation learning, diffusion and flow policies, world models, and generalization.** I'm sharing the filtered list here in case it's useful to others at the conference.

Papers are grouped into **P0 (must read, 40 papers)**, **P1 (should read, 85)**, and **P2 (nice to read, 46)**. Each entry has a one-line summary written from the abstract and the paper's session time and room.
- Where a paper has an arXiv preprint (✅), I've linked it directly.
- RA-L papers presented at IROS link to IEEE Xplore.
- The IROS 2026 proceedings aren't on IEEE Xplore yet. For the remaining papers, the IEEE Xplore link is a title search that will find the paper once the proceedings are posted.

<!--more-->

## P0 — Must Read

### VLA Models
- ✅ **Enhancing Generalization in Vision–Language–Action Models by Preserving Pretrained Representations** — Grover, Gopalkrishnan, Ai et al. — [arXiv:2509.11417](https://arxiv.org/abs/2509.11417) — *Wed 14:52, Room 411/412*  
  Preserves VLM representations in VLA fine-tuning via frozen+trainable dual encoders, string action tokens, and VL co-training, improving generalization.
- ✅ **EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos** — Yang, Yu, Wu et al. — [arXiv:2507.12440](https://arxiv.org/abs/2507.12440) — *Tue 15:19, Room 317/318*  
  Trains a VLA on egocentric human video predicting wrist/hand actions, retargets to robots and fine-tunes with few demos, ablating human data value.
- ✅ **Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos** — XU, Zhang, Wang et al. — [arXiv:2606.18955](https://arxiv.org/abs/2606.18955) — *Tue 14:33, Room 317/318*  
  Disentangled VQ-VAE latent actions extract motion priors from unlabeled human egocentric video for cross-embodiment VLA pretraining.
- ✅ **Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions** — Feng, Jain — [arXiv:2608.03563](https://arxiv.org/abs/2608.03563) — *Tue 15:48, Room 317/318*  
  Changes the VLA prediction target to a unified latent encoding motor control and visual scene transitions, improving training efficiency without architecture changes.
- ✅ **Recursive Belief Vision Language Action Models** — Bagaria, Patel, Sebastian — [arXiv:2602.20659](https://arxiv.org/abs/2602.20659) — *Tue 15:45, Room 317/318*  
  VLA with a recursive latent belief trained via world-model objectives for partial observability; big long-horizon gains over pi0.
- ✅ **MaskVLA: Visual Masking against Trajectory Overfitting of Vision-Language-Action Model** — Jiang, Huang, Wang et al. — [arXiv:2609.23565](https://arxiv.org/abs/2609.23565) — *Mon 9:55, Room 411/412*  
  Finds VLAs severely overfit trajectories when fine-tuned on small data; random main-camera masking forces use of wrist-view features.

### RL Fine-Tuning & Policy Optimization
- ✅ **Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models** — Liu, Zhang, Liu et al. — [arXiv:2607.16506](https://arxiv.org/abs/2607.16506) — *Mon 9:19, Room 409/410*  
  Residual RL over frozen VLA with learned foresight value rewards shapes subtask handoff states, raising chained-task success from 54% to 86%.
- ✅ **AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models** — Sun, XU, Cao et al. — [arXiv:2603.08519](https://arxiv.org/abs/2603.08519) — *Tue 10:09, Room 302/303*  
  Decomposes demos into LLM-generated subtasks and uses a latent world model to score action chunks, enabling offline GRPO post-training of VLAs.
- ✅ **VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning** — Lu, Guo, Zhang et al. — [arXiv:2505.18719](https://arxiv.org/abs/2505.18719) — *Mon 9:29, Room 409/410*  
  Online RL framework for autoregressive VLAs with VLM process reward model and scaling tricks, improving OpenVLA on LIBERO and real robots.
- ✅ **Beyond Imitation: Reinforcement Learning Fine-Tuning for Adaptive Diffusion Navigation Policies** — Sheng, Bai, Xu et al. — [arXiv:2603.12868](https://arxiv.org/abs/2603.12868) — *Wed 9:06, Room 409/410*  
  Fine-tunes diffusion navigation policies with GRPO over sampled trajectories, avoiding value networks; improves OOD safety and transfers to real.
- ✅ **Human-In-The-Loop Online Rejection Sampling for Robotic Manipulation** — Lu, Zhao, Lin et al. — [arXiv:2510.26406](https://arxiv.org/abs/2510.26406) — *Mon 10:24, Room 411/412*  
  Human-in-the-loop online rejection sampling with reward-weighted supervision fine-tunes pi0 stably in 1.5h, beating RL baselines.
- ✅ **Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation** — Tang, Jing, Pan et al. — [arXiv:2505.12744](https://arxiv.org/abs/2505.12744) — *Wed 9:41, Room 403/404*  
  LMM directly predicts gripper goals via reasoning; SFT on reasoning data then RL in simulation improves OOD generalization and sim-to-real.
- ✅ **TD-GRPC: Temporal Difference Learning with Group Relative Policy Constraint for Humanoid Locomotion** — Nguyen, Le, Nguyen et al. — [arXiv:2505.13549](https://arxiv.org/abs/2505.13549) — *Mon 9:18, Room 315/316*  
  Combines TD-MPC with GRPO-style group-relative ranking and latent trust-region policy constraints for stable humanoid RL.

### Robot Learning & Imitation
- **What Can Robot Foundation Models Learn from Limited Human Demonstrations? Fine-Tuning with Human Task Videos** — Hagane, Goto, Ohama — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22What%20Can%20Robot%20Foundation%20Models%20Learn%20from%20Limited%20Human%20Demonstrations%3F%20Fine-Tuning%20with%20Human%20Task%20Videos%22) — *Mon 9:33, Room 329*  
  Systematic study mixing human task videos into VLA post-training; helps semantic grounding of novel objects but not contact-rich verb-specific motions.
- ✅ **CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations** — Liang, Czempin, Hong et al. — [arXiv:2505.04999](https://arxiv.org/abs/2505.04999) — *Mon 14:39, Room 302/303*  
  Continuous latent action model learned from action-free demos, grounded with a jointly trained decoder on play data, approaching BC with true labels.
- ✅ **Geometric Entropy: When Trajectory Diversity Helps and Hurts in Imitation Learning** — Luo, Liu, Zhou et al. — [arXiv:2606.20871](https://arxiv.org/abs/2606.20871) — *Mon 15:11, Room 302/303*  
  Geometric entropy metric reveals an inverted-U between demonstration trajectory diversity and imitation success, shifting with data scale and pretrained VLAs.
- ✅ **Learning from the Best: Smoothness-Driven Metrics for Data Quality in Imitation Learning** — Kulkarni, Dhar, Cui — [arXiv:2604.23000](https://arxiv.org/abs/2604.23000) — *Mon 15:25, Room 302/303*  
  Smoothness metrics score demonstrations, reducing conditional action variance; filtering gives +16% BC success with one-sixth of the data and helps retrieval/reweighting.
- **Improving Policy Learning from Delayed Interventions** — Zhu, Dudek, Meger — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Improving%20Policy%20Learning%20from%20Delayed%20Interventions%22) — *Mon 15:16, Room 329*  
  Shows delayed human interventions hurt interactive imitation and learns an intervention-based cost function accounting for delay to recover performance.
- **TRACE: Trajectory Action Chunking Enhancement for Robust Imitation Learning in Robotic Manipulation** — Hu, Luo, Liu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22TRACE%3A%20Trajectory%20Action%20Chunking%20Enhancement%20for%20Robust%20Imitation%20Learning%20in%20Robotic%20Manipulation%22) — *Tue 14:48, Room 317/318*  
  Finds ACT degrades with larger biased datasets due to spurious correlations; debiasing encoder via action masking/noise fixes it.
- ✅ **Beyond Implicit Force: Evaluating Explicit Force-Torque Proxies in Action Chunking with Transformers** — Wong, Liu, Dayoub — [arXiv:2607.14578](https://arxiv.org/abs/2607.14578) — *Mon 9:09, Room 411/412*  
  Shows ACT's force-awareness comes from leader-follower teleop discrepancy; torque proxies recover contact-aware behavior.

### Diffusion & Flow Policies
- ✅ **Amortizing Trajectory Diffusion with Keyed Drift Fields** — Puthumanaillam, Ornik — [arXiv:2603.14056](https://arxiv.org/abs/2603.14056) — *Mon 9:00, Room 321*  
  One-step trajectory generator trained with a drift-field objective in a condition-aware key space, matching diffusion planning at far lower latency.
- ✅ **VGFM: Expressive Robot Policies Via Dense Value Guidance in Flow Matching** — Koirala, Campbell — [arXiv:2609.14261](https://arxiv.org/abs/2609.14261) — *Tue 14:41, Room 320*  
  Dense value guidance along flow-matching trajectories enables offline RL policy improvement without BPTT or distillation.
- **Compositional Average Velocity Modeling for Efficient Diffusion Offline Reinforcement Learning** — Nguyen, Syarubany, Kim et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Compositional%20Average%20Velocity%20Modeling%20for%20Efficient%20Diffusion%20Offline%20Reinforcement%20Learning%22) — *Mon 9:23, Room 409/410*  
  Single-step flow policy for offline RL via compositional average-velocity modeling with behavior-regularized actor-critic, no BPTT or distillation.
- **ExplicitDP: Generalized Multi-Task Manipulation with Explicit Diffusion Policies Via Phase-Aware Task Understanding** — Fang, Tao, Li — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22ExplicitDP%3A%20Generalized%20Multi-Task%20Manipulation%20with%20Explicit%20Diffusion%20Policies%20Via%20Phase-Aware%20Task%20Understanding%22) — *Wed 15:25, Room 411/412*  
  Shows multi-task diffusion policies suffer phase confusion as latents organize by visual similarity; adds explicit phase-aware task structure.

### World Models
- ✅ **Generalizable Robotic Insertion with World Models** — Hansen, Akinola, Guo et al. — [arXiv:2609.28258](https://arxiv.org/abs/2609.28258) — *Mon 9:18, Room 411/412*  
  A single world model trained on up to 90 insertion tasks generalizes zero-shot to unseen parts (56% vs 7% model-free) and scales with objects.
- ✅ **ImagiNav: Scalable Embodied Navigation Via Generative Visual Prediction and Inverse Dynamics** — Chen, Cai, Wang et al. — [arXiv:2603.13833](https://arxiv.org/abs/2603.13833) — *Mon 15:25, Room 401/402*  
  Navigation via language-conditioned egocentric video generation plus an inverse dynamics model, enabling zero-shot transfer from in-the-wild videos without robot demos.
- ✅ **Scaling Cross-Embodiment World Models for Dexterous Manipulation** — He, Ai, Mu et al. — [arXiv:2511.01177](https://arxiv.org/abs/2511.01177) — *Wed 9:34, Room 304/305*  
  Particle-based shared action/state representation lets world models scale across human and robot hands for cross-embodiment dexterous control.
- ✅ **GrndCtrl: Grounding World Models Via Self-Supervised Reward Alignment** — He, Patrikar, Kim et al. — [arXiv:2512.01952](https://arxiv.org/abs/2512.01952) — *Wed 9:46, Room 409/410*  
  RLVR-style self-supervised post-training aligns pretrained video world models with geometric and perceptual verifiable rewards.
- **RoDyn: Taming Interactive Robot-Dynamic 2.5D World Model for Robotic Manipulation** — Zhang, Wu, Lu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22RoDyn%3A%20Taming%20Interactive%20Robot-Dynamic%202.5D%20World%20Model%20for%20Robotic%20Manipulation%22) — *Wed 10:04, Room 409/410*  
  2.5D robot-dynamic world model with geometry-aware tokenizer; improves generation fidelity and accelerates MBRL and imitation learning.
- ✅ **DAWN: Noise-Robust Quadruped Parkour Via Depth-Denoising World Models** (Award Candidate) — Choi, Kim, Kim et al. — [arXiv:2609.29092](https://arxiv.org/abs/2609.29092) — *Mon 15:16, Room 406*  
  World model trained with noisy-to-clean depth reconstruction and contrastive alignment enables filter-free quadruped parkour.

### Generalization, Sim-to-Real & Evaluation
- ✅ **How VLAs (Really) Work in Open-World Environments** — Rasouli, Wu, Li et al. — [arXiv:2604.21192](https://arxiv.org/abs/2604.21192) — *Tue 15:19, Room 302/303*  
  Analyzes SOTA VLAs on BEHAVIOR-1K, showing success-rate metrics hide safety violations and inconsistency; proposes safety-aware evaluation protocols.
- ✅ **Abstract Sim2Real through Approximate Information States** — Deng, Li, Hanna — [arXiv:2604.15289](https://arxiv.org/abs/2604.15289) · [RA-L](https://doi.org/10.1109/lra.2026.3688068) — *Tue 14:45, Room 409/410*  
  Formalizes sim2real with abstract simulators via state abstraction; shows history-conditioned grounding with real data enables transfer.
- ✅ **REALM: A Real-To-Sim Validated Benchmark for Generalization in Robotic Manipulation** — Sedlacek, Yefanov, Ponimatkin et al. — [arXiv:2512.19562](https://arxiv.org/abs/2512.19562) · [RA-L](https://doi.org/10.1109/lra.2026.3692088) — *Wed 15:04, Room 317/318*  
  Real-to-sim validated benchmark with 15 perturbation factors showing pi0, pi0-FAST, GR00T generalization gaps with sim-real correlation.
- ✅ **Spotlighting Task-Relevant Features: Object-Centric Representations for Better Generalization in Robotic Manipulation** — Chapin, Machado, Dellandrea et al. — [arXiv:2601.21416](https://arxiv.org/abs/2601.21416) — *Wed 9:11, Room 329*  
  Large diagnostic study: slot-based object-centric representations greatly improve policy robustness to visual shifts but merge slots under clutter.
- ✅ **The Moving Eye: Enhancing VLA Spatial Generalization Via Hybrid Dynamic Data Collection** — Tang, Zhu, Xie et al. — [arXiv:2607.02322](https://arxiv.org/abs/2607.02322) — *Mon 9:26, Room 411/412*  
  Shows VLAs learn spatial shortcuts from fixed camera poses; dynamic moving-camera data collection improves spatial generalization.
- ✅ **LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models** — Hou, Zhao — [arXiv:2603.00592](https://arxiv.org/abs/2603.00592) — *Mon 10:12, Room 401/402*  
  Shows SOTA VLAs like pi0.5 largely ignore language; builds a semantic-perturbation benchmark and tests augmentation to close the gap.
- ✅ **On the Generalization Capabilities, Design Choices and Limitations of Keypoint Imitation Learning** — Lips, Moletta, Welle et al. — [arXiv:2605.26649](https://arxiv.org/abs/2605.26649) — *Mon 14:30, Room 302/303*  
  Systematic real-world study (2000+ rollouts) of keypoint imitation design choices, generalization, and limits versus other representations.
- ✅ **BIFROST: Bridging Invariant Feature Representation for Observation-Space Sim2Real Transfer** — Deng, Hanna — [arXiv:2607.01410](https://arxiv.org/abs/2607.01410) — *Mon 15:49, Room 334*  
  Cross-domain bisimulation history encoder on paired sim/real data enables zero-shot sim2real under both visual and dynamics gaps.

### Reward Learning & Foundation-Model Rewards
- **IRL4IDM: Inverse Reinforcement Learning for Inverse Dynamics Model in Video Generative Robot Planner** — Huang, Zhu, Ren et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22IRL4IDM%3A%20Inverse%20Reinforcement%20Learning%20for%20Inverse%20Dynamics%20Model%20in%20Video%20Generative%20Robot%20Planner%22) — *Tue 15:31, Room 409/410*  
  Learns inverse dynamics for video-generative planners via IRL: dense rewards from video temporal alignment and online goal-conditioned RL, no action labels.
- ✅ **Generalizable Dense Reward for Long-Horizon Robotic Tasks** — Yong, Sheng, Qi et al. — [arXiv:2604.00055](https://arxiv.org/abs/2604.00055) — *Mon 9:22, Room 320*  
  Dense rewards for RL fine-tuning foundation policies: VLM progress for value init plus self-certainty intrinsic reward, beating RL fine-tuning baselines.

## P1 — Should Read

### VLA Models
- ✅ **ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning** — Yang, Shi, Mingyuan et al. — [arXiv:2607.12992](https://arxiv.org/abs/2607.12992) — *Mon 9:49, Room 411/412*  
  Seam-aware training for chunked VLA policies with continuity losses, history corruption, and AWAC fine-tuning to reduce chunk-boundary jitter.
- ✅ **FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models** — Lin, Duan, Fang et al. — [arXiv:2510.01642](https://arxiv.org/abs/2510.01642) — *Mon 15:51, Room 411/412*  
  Automatically generates failure cases with executable recovery actions; a VLM trained on them improves pi0-FAST/OpenVLA recovery by up to 22.6%.
- ✅ **HapticVLA: Contact-Rich Manipulation Via Vision-Language-Action Model without Inference-Time Tactile Sensing** — Gubernatorov, Sannikov, Mikhalchuk et al. — [arXiv:2603.15257](https://arxiv.org/abs/2603.15257) — *Mon 15:39, Room 302/303*  
  Safety-aware reward-weighted flow matching with tactile rewards, then distills a tactile token so VLA needs no tactile sensors at inference.
- ✅ **LLaDA-VLA: Vision Language Diffusion Action Models** — Wen, Li, Gu et al. — [arXiv:2509.06932](https://arxiv.org/abs/2509.06932) — *Wed 14:37, Room 411/412*  
  Builds a VLA on masked-diffusion VLMs with action-token classification and hierarchical decoding, outperforming autoregressive VLAs.
- ✅ **GeoVLA: Empowering 3D Representations in Vision-Language-Action Models** (Award Candidate) — Sun, Xie, Liu et al. — [arXiv:2508.09071](https://arxiv.org/abs/2508.09071) — *Tue 9:33, Room 406*  
  Adds point-cloud embeddings and a 3D-aware action expert to a VLA, improving LIBERO/ManiSkill2 and viewpoint robustness.
- **BeliefVLA: Implicit Belief-State Driven Vision-Language-Action Models for Robust Manipulation** — Liu, Huang, Han et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22BeliefVLA%3A%20Implicit%20Belief-State%20Driven%20Vision-Language-Action%20Models%20for%20Robust%20Manipulation%22) — *Wed 15:01, Room 411/412*  
  Injects V-JEPA2 implicit belief states from history into VLAs to add dynamics-aware priors and temporal memory.

### RL Fine-Tuning & Policy Optimization
- ✅ **Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning** — Zhang, Weng, Liu et al. — [arXiv:2607.13818](https://arxiv.org/abs/2607.13818) — *Mon 9:12, Room 411/412*  
  Agentic RL policy monitors VLA execution quality metrics and selects recovery modes, boosting LIBERO robustness under disturbances.
- **DexHiL: A Human-In-The-Loop Framework for Post-Training VLA Models in Dexterous Manipulation** — Han, Chen, Xu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22DexHiL%3A%20A%20Human-In-The-Loop%20Framework%20for%20Post-Training%20VLA%20Models%20in%20Dexterous%20Manipulation%22) — *Tue 15:49, Room 328*  
  Human-in-the-loop arm-hand intervention framework with intervention-aware sampling for post-training dexterous VLAs, +25% over offline.
- ✅ **HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual Reinforcement Learning** — Liu, Mai, He et al. — [arXiv:2606.22860](https://arxiv.org/abs/2606.22860) — *Tue 15:34, Room 409/410*  
  Model-agnostic residual RL adapter on VLA actions with human-in-the-loop guidance reaches >95% real success in 1.5 hours.
- ✅ **NavCMPO: Critic-Guided MeanFlow Policy Optimization for Adaptive Navigation** — An, Wu, Liu et al. — [arXiv:2607.14643](https://arxiv.org/abs/2607.14643) — *Wed 9:03, Room 409/410*  
  Few-step MeanFlow navigation policy with critic-gradient trajectory refinement, then PPO fine-tuning with BC regularization for embodiment adaptation.
- ✅ **Keyframe-Guided Structured Rewards for Reinforcement Learning in Long-Horizon Laboratory Robotics** — Qiu, Sun, Ye et al. — [arXiv:2603.00719](https://arxiv.org/abs/2603.00719) — *Tue 15:16, Room 409/410*  
  Keyframe-derived progress rewards drive human-in-the-loop RL fine-tuning of a VLA, reaching 82% on lab tasks vs 42-47% for HG-DAgger/ConRFT.
- **HIL-ResRFT: Improving Imitation Learning Policy with Human in the Loop Residual RL for Robot Manipulation** — Li, Zhou, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22HIL-ResRFT%3A%20Improving%20Imitation%20Learning%20Policy%20with%20Human%20in%20the%20Loop%20Residual%20RL%20for%20Robot%20Manipulation%22) — *Tue 9:03, Room 302/303*  
  Human-in-the-loop residual RL fine-tunes imitation policies in the real world for robustness and sample efficiency.

### Robot Learning & Imitation
- **Trajectory-Based Diffusion from One‑shot Human Video for Generalizable Manipulation** — Pan, Yu, Cao et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Trajectory-Based%20Diffusion%20from%20One%E2%80%91shot%20Human%20Video%20for%20Generalizable%20Manipulation%22) — *Mon 14:36, Room 302/303*  
  Extracts object 6D trajectories from one-shot human video as embodiment-agnostic representation to generate data and train trajectory transformer.
- ✅ **Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition** — Quartey, Castro, Rosen et al. — [arXiv:2605.20648](https://arxiv.org/abs/2605.20648) — *Mon 10:01, Room 411/412*  
  Policies jointly generating actions and predicate beliefs improve both and enable zero-shot skill composition via symbolic planning.
- ✅ **Canonical Policy: Learning Canonical 3D Representation for SE(3)-Equivariant Policy** — Zhang, Xu, Lakamsani et al. — [arXiv:2505.18474](https://arxiv.org/abs/2505.18474) — *Wed 9:00, Room 329*  
  Canonical 3D point-cloud representation theory yields SE(3)-equivariant generative imitation policies generalizing to new poses/views.
- ✅ **FlowCorrect: Efficient Interactive Correction of Generative Flow Policies for Robotic Manipulation** — Welte, Shi, Wolf et al. — [arXiv:2602.22056](https://arxiv.org/abs/2602.22056) — *Wed 9:33, Room 329*  
  Adapts flow-matching policies at deployment from sparse human VR pose nudges without retraining, fixing near-miss failures.
- **Mitigating Perceptual Interference in Multi-Stage Visuomotor Imitation with Stage-Conditioned Spatial Gating** — Zhang, Wang, Chen — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Mitigating%20Perceptual%20Interference%20in%20Multi-Stage%20Visuomotor%20Imitation%20with%20Stage-Conditioned%20Spatial%20Gating%22) — *Tue 10:19, Room 329*  
  Identifies perceptual interference in multi-stage imitation and adds stage-conditioned spatial gating in the encoder, improving late-stage success.
- ✅ **From Code to Action: Hierarchical Learning of Diffusion-VLM Policies** — Peschl, Mazzaglia, Dijkman — [arXiv:2509.24917](https://arxiv.org/abs/2509.24917) — *Mon 10:24, Room 401/402*  
  Uses robot API subtask functions as supervision: a code-generating VLM decomposes tasks grounded by diffusion policies with memory.
- **Augmenting Robot Imitation Learning with Human Videos without Additional Robot Data** — Zhang, Xiong, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Augmenting%20Robot%20Imitation%20Learning%20with%20Human%20Videos%20without%20Additional%20Robot%20Data%22) — *Mon 10:19, Room 329*  
  Augments fixed-budget robot imitation with unpaired human videos via hand-keypoint to joint-space mapping and cross-domain feature alignment.
- ✅ **RoboSSM: Scalable In-Context Imitation Learning Via State-Space Models** — Yoo, Hu, Zhu et al. — [arXiv:2509.19658](https://arxiv.org/abs/2509.19658) — *Mon 9:35, Room 409/410*  
  Replaces Transformers with state-space models for in-context imitation learning, extrapolating to longer demonstration prompts.
- ✅ **Generalizable Hierarchical Skill Learning Via Object-Centric Representation** — Zhao, Qi, Hu et al. — [arXiv:2510.21121](https://arxiv.org/abs/2510.21121) — *Mon 14:59, Room 302/303*  
  Object-centric canonicalized skill primitives bridge a VLM planner and low-level visuomotor policy for generalization.
- ✅ **Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning** — Krohn, Prasad, Tiboni et al. — [arXiv:2511.14427](https://arxiv.org/abs/2511.14427) · [RA-L](https://doi.org/10.1109/lra.2026.3681156) — *Mon 9:57, Room 320*  
  Masked multisensory autoencoder pretraining gives robust representations for contact-rich RL.
- ✅ **ICLR: In-Context Imitation Learning with Visual Reasoning** — Nguyen, Yuan, Wei et al. — [arXiv:2603.07530](https://arxiv.org/abs/2603.07530) — *Mon 9:22, Room 329*  
  In-context imitation learning that also generates visual reasoning traces (future trajectories in image space), improving generalization to unseen tasks.
- ✅ **SAIL: Test-Time Scaling for In-Context Imitation Learning with VLM** — Sato, Iwasawa, Tang et al. — [arXiv:2603.08269](https://arxiv.org/abs/2603.08269) — *Tue 15:49, Room 320*  
  Test-time scaling for in-context imitation via MCTS over trajectory refinements scored by a VLM.

### Diffusion & Flow Policies
- **SANE: Smooth Action Via Noise Envelope Conditioned on Gaussian Process for Denoising Policies** — Lv, Tan, Zhao et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22SANE%3A%20Smooth%20Action%20Via%20Noise%20Envelope%20Conditioned%20on%20Gaussian%20Process%20for%20Denoising%20Policies%22) — *Wed 15:31, Room 411/412*  
  Replaces i.i.d. flow-matching noise with a Gaussian-process noise prior conditioned on executed actions for smooth inter-chunk VLA actions.
- **Learning Consistent Manipulation Via Joint Action–Object Diffusion** — Yu, Fan — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Learning%20Consistent%20Manipulation%20Via%20Joint%20Action%E2%80%93Object%20Diffusion%22) — *Tue 9:58, Room 302/303*  
  Diffusion policy jointly denoising robot and object trajectories enforces action-outcome consistency and improves manipulation robustness.
- ✅ **SUREFlow: State-Space Uncertainty-Aware REsidual Flow Matching for Robust Robot Manipulation** — Islam, Peddapalli, Lee et al. — [arXiv:2607.10504](https://arxiv.org/abs/2607.10504) — *Wed 14:52, Room 328*  
  Residual flow-matching policy that predicts input-dependent uncertainty to selectively refine unreliable action dimensions, strong LIBERO results at 179M params.
- ✅ **HiFlow: Tokenization-Free Scale-Wise Autoregressive Policy Learning Via Flow Matching** — Yashima, Seno, Kurita et al. — [arXiv:2603.27281](https://arxiv.org/abs/2603.27281) — *Mon 9:35, Room 411/412*  
  Tokenization-free coarse-to-fine autoregressive flow policy using temporally pooled multi-scale action targets, beating diffusion and tokenized AR policies.
- ✅ **RoamFlow: Reinforcement-Aligned One-Step Action MeanFlow Policy for Image-Goal Navigation** — ZHANG, Chen, Gao et al. — [arXiv:2606.29934](https://arxiv.org/abs/2606.29934) — *Wed 9:00, Room 409/410*  
  One-step MeanFlow navigation policy initialized by imitation then refined with RL for image-goal navigation.
- ✅ **Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration** — Miller, Jagersand — [arXiv:2607.29482](https://arxiv.org/abs/2607.29482) — *Mon 15:31, Room 302/303*  
  Stochastic-interpolant policy initialized from recent action history instead of Gaussian noise, cutting transport cost ~10x and latency while matching success.
- ✅ **CoRDE: Concept-Prior Routed Diffusion Experts for Structural Generalization in Robot Manipulation** — Huang, Zhao, Zhou et al. — [arXiv:2606.21935](https://arxiv.org/abs/2606.21935) — *Mon 14:33, Room 302/303*  
  Concept-prior routed LoRA diffusion experts avoid MoE routing collapse and gradient conflict in multi-task long-horizon imitation.
- ✅ **Trajectory-Consistent Flow Matching for Robust Visuomotor Policy Learning** — Ahmed, Nag, Akash et al. — [arXiv:2605.08511](https://arxiv.org/abs/2605.08511) — *Mon 9:38, Room 409/410*  
  Closes flow-matching train-inference gap with multi-step trajectory consistency training, velocity smoothness regularization and RK4 integration.
- ✅ **3D FlowMatch Actor: Unified 3D Policy for Single and Dual-Arm Manipulation** — Gkanatsios, Xu, Bronars et al. — [arXiv:2508.11002](https://arxiv.org/abs/2508.11002) — *Tue 10:15, Room 302/303*  
  3D flow-matching policy with relative attention achieving 30x faster training/inference and large SOTA gains on bimanual PerAct2 and RLBench.
- ✅ **FLUX: Accelerating Cross-Embodiment Generative Navigation Policies Via Rectified Flow and Static-To-Dynamic Learning** — Gong, Zhong, DING et al. — [arXiv:2603.12806](https://arxiv.org/abs/2603.12806) — *Wed 9:49, Room 409/410*  
  Rectified-flow navigation policy pretrained then RL-refined in crowds; stochastic RL-learned recovery aids generalization, zero-shot across wheeled, quadruped, humanoid.
- ✅ **D3P: Dynamic Denoising Diffusion Policy Via Reinforcement Learning** — Yu, Gao, Wu et al. — [arXiv:2508.06804](https://arxiv.org/abs/2508.06804) — *Tue 15:27, Room 320*  
  RL-trained adapter allocates denoising steps per action at test time, exploiting that only some actions are critical.
- ✅ **From Flow to One Step: Real-Time Multi-Modal Trajectory Policies Via Implicit Maximum Likelihood Estimation-Based Distribution Distillation** — Dong, Zhang, Zhang et al. — [arXiv:2603.09415](https://arxiv.org/abs/2603.09415) — *Wed 15:22, Room 411/412*  
  Distills flow-matching policy into a one-step student via IMLE with Chamfer set loss, preserving multimodality at 125 Hz.

### World Models
- ✅ **RoboDream: Compositional World Models for Scalable Robot Data Synthesis** — Ye, Xue, Van Hoorick et al. — [arXiv:2606.02577](https://arxiv.org/abs/2606.02577) — *Wed 15:51, Room 411/412*  
  Embodiment-centric video world model synthesizes demonstrations in new scenes/objects/viewpoints, improving downstream policies with less real data.
- ✅ **DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation Via World Model** — Yin, Lai — [arXiv:2608.22278](https://arxiv.org/abs/2608.22278) — *Mon 15:11, Room 401/402*  
  Uses an RSSM world model as predictive representation and multi-step supervision to distill privileged teachers into visual humanoid loco-manipulation policies.
- **DreamManip: Holistic Visual Planning for Long-Horizon Robotic Manipulation via Video Foundation Models** — Xiong, Zhong, Fan et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22DreamManip%3A%20Holistic%20Visual%20Planning%20for%20Long-Horizon%20Robotic%20Manipulation%20via%20Video%20Foundation%20Models%22) — *Wed 15:45, Room 411/412*  
  Video foundation model imagines the full visual path from current to goal image, then executes it for long-horizon manipulation.
- ✅ **Cosmos-Surg-dVRK: World Foundation Model-Based Automated Online Evaluation of Surgical Robot Policy Learning** — Zbinden, Nelson, Chen et al. — [arXiv:2510.16240](https://arxiv.org/abs/2510.16240) · [RA-L](https://doi.org/10.1109/lra.2026.3675962) — *Tue 9:07, Room 409/410*  
  Fine-tunes the Cosmos world foundation model to evaluate surgical robot policies online in place of physical rollouts.

### Generalization, Sim-to-Real & Evaluation
- ✅ **Factor-Aware Mixture-Of-Experts with Pretrained Encoder for Combinatorial Generalization** — Zhang, Zhan, He et al. — [arXiv:2606.21100](https://arxiv.org/abs/2606.21100) — *Wed 9:52, Room 315/316*  
  Factor-specific adapters routed by a mixture-of-experts on a frozen encoder give diffusion policies combinatorial generalization to lighting/texture shifts.
- ✅ **AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models** — Heo, Woo, Kim et al. — [arXiv:2603.05868](https://arxiv.org/abs/2603.05868) — *Mon 10:09, Room 401/402*  
  Zero-shot viewpoint robustness for VLAs by re-rendering test camera views to the training viewpoint with feed-forward novel view synthesis.
- ✅ **Attentive Feature Aggregation Or: How Policies Learn to Stop Worrying about Robustness and Attend to Task-Relevant Visual Cues** — Tsagkas, Sochopoulos, Danier et al. — [arXiv:2511.10762](https://arxiv.org/abs/2511.10762) — *Mon 9:29, Room 411/412*  
  Trainable attentive pooling over pretrained visual features makes visuomotor policies robust to distractors without augmentation or PVR fine-tuning.
- **Scene2Gym: Transforming Real-World Scenes into Interactive Training Environments for Robot Learning** — Ji, Chen, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Scene2Gym%3A%20Transforming%20Real-World%20Scenes%20into%20Interactive%20Training%20Environments%20for%20Robot%20Learning%22) — *Wed 9:06, Room 411/412*  
  Builds metric-aligned 3DGS digital twins from real scenes; IL and visual RL policies trained there transfer zero-shot to real.
- ✅ **GAP: Geometric Anchor Pre-Training for Data-Efficient Visuomotor Learning of Manipulation Tasks** — Buoso, Protopapa, Di Carlo et al. — [arXiv:2605.15836](https://arxiv.org/abs/2605.15836) — *Wed 10:00, Room 329*  
  Action-free pretraining of spatial pooling adapter on simulated object masks yields geometric anchors for data-efficient, robust few-shot IL.
- ✅ **ManiVID-3D: Generalizable View-Invariant Reinforcement Learning for Robotic Manipulation Via Disentangled 3D Representations** — Li, Qu, Jia et al. — [arXiv:2509.11125](https://arxiv.org/abs/2509.11125) · [RA-L](https://doi.org/10.1109/lra.2026.3662647) — *Wed 15:49, Room 328*  
  View-invariant 3D visual RL via disentangled features and calibration-free point cloud alignment; +40% under viewpoint shifts.
- ✅ **DADiff: Diffusion-Driven Cross-Domain Policy Adaptation for Reinforcement Learning** — Chen, Satheesh, Da et al. — [arXiv:2607.16090](https://arxiv.org/abs/2607.16090) — *Mon 9:11, Room 320*  
  Uses diffusion generative trajectory discrepancies to estimate dynamics mismatch for reward modification or data selection in cross-domain RL, with bounds.
- ✅ **ForesightSafety-VLA: A Unified Diagnostic Safety Benchmark for Vision-Language-Action Models** — Lyu, Sun, Jia et al. — [arXiv:2606.27079](https://arxiv.org/abs/2606.27079) — *Mon 15:45, Room 411/412*  
  Diagnostic VLA safety benchmark with 13-category taxonomy and process-level risk metrics, finding perception/structure shifts degrade safety most.
- ✅ **ZeroBot: Learning from Scratch in Minutes with Generative Real2Sim** — Kapelyukh, Zhang, James et al. — [RA-L](https://doi.org/10.1109/lra.2026.3662595) — *Tue 10:24, Room 401/402*  
  Image-to-3D generative real2sim plus contact-sampling action space enables learning real manipulation from scratch via parallel RL in minutes.
- ✅ **Red-Teaming Vision-Language-Action Models Via Quality Diversity Prompt Generation for Robust Robot Policies** — Srikanth, Liang, Hsu et al. — [arXiv:2603.12510](https://arxiv.org/abs/2603.12510) — *Tue 9:33, Room 329*  
  Quality-diversity red-teaming generates diverse task-relevant instructions that make VLAs fail, then uses them to improve robustness.
- **SRDR: Smoothed Return-Guided Domain Randomization for Stable and Robust Policy Learning** — Wang, Ghoreishi — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22SRDR%3A%20Smoothed%20Return-Guided%20Domain%20Randomization%20for%20Stable%20and%20Robust%20Policy%20Learning%22) — *Mon 14:42, Room 411/412*  
  Variance-reduced smoothed-return adaptive domain randomization for stable, robust policy learning across seeds.
- ✅ **Structure-Aware Robust Fine-Tuning: Defending Vision-Language-Action Robots against Physical Attention Hijacking** — Zhang, Yin, Yang et al. — [arXiv:2608.03231](https://arxiv.org/abs/2608.03231) — *Mon 15:39, Room 411/412*  
  Shows printable adversarial patches hijack VLA action-to-vision attention; proposes structure-aware visual-encoder fine-tuning defense.

### Reward Learning & Foundation-Model Rewards
- ✅ **Training Fast Robot Policies with Slow Foundation Models** — Singh, Bhattacharyya, Namboodiri et al. — [arXiv:2406.05881](https://arxiv.org/abs/2406.05881) — *Tue 15:38, Room 320*  
  LLM-written reward code refined by VLM diagnosis of failed rollouts trains fast policies without foundation models at deployment.
- ✅ **MotionVL: Vision-Language Supervision for Reinforcement Learning of Humanoid Motion** — Luo, Wu, Xiong et al. — [RA-L](https://doi.org/10.1109/lra.2026.3669793) — *Mon 15:05, Room 401/402*  
  VLM describes humanoid behavior and LLM generates/refines rewards in closed loop for humanoid motion RL.
- **Warming-Up and Shaping Robot Policy Learning with Large Language Model-Based Critic Feedback** — Hayamizu, Ai, hou et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Warming-Up%20and%20Shaping%20Robot%20Policy%20Learning%20with%20Large%20Language%20Model-Based%20Critic%20Feedback%22) — *Tue 15:19, Room 409/410*  
  Uses LLMs as external critics to initialize and shape value estimates, improving RL sample efficiency over LLM-generated policies.
- **SPEAR: Selective Preference Elicitation with Adaptive Rewards for VLM-Guided Robotic Manipulation** — Wu, Liu, Zhou et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22SPEAR%3A%20Selective%20Preference%20Elicitation%20with%20Adaptive%20Rewards%20for%20VLM-Guided%20Robotic%20Manipulation%22) — *Mon 10:15, Room 411/412*  
  Makes VLMs efficient preference oracles via distinguishability-aware query selection and adaptive reward transition, learning real Mobile ALOHA tasks online.
- ✅ **Stage-Transition Dense Reward Modeling for Reinforcement Learning** — yang, Chen, Wang et al. — [arXiv:2606.31377](https://arxiv.org/abs/2606.31377) — *Tue 15:28, Room 409/410*  
  Converts expert videos into stage-transition and within-stage progress dense rewards with OOD detection to train RL from scratch.
- ✅ **PrefMoE: Robust Preference Modeling with Mixture-Of-Experts Reward Learning** — Yuan, Wang, Zhao et al. — [arXiv:2605.00384](https://arxiv.org/abs/2605.00384) — *Tue 10:18, Room 403/404*  
  Mixture-of-experts reward model with soft trajectory routing handles heterogeneous, conflicting preference labels for more robust preference-based RL.
- ✅ **Learning Acrobatic Flight from Preferences** — Merk, Geles, Xing et al. — [arXiv:2508.18817](https://arxiv.org/abs/2508.18817) — *Tue 14:34, Room 315/316*  
  Distributional reward ensembles model per-step uncertainty in preference-based RL, reaching 88% of shaped-reward performance for real acrobatic flight.

### RL Algorithms & Exploration
- **Escaping the Greedy Trap: Robust Manipulation Via Spatial Beam Search in Coarse-To-Fine Reinforcement Learning** — Xiang, Niu, Gu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Escaping%20the%20Greedy%20Trap%3A%20Robust%20Manipulation%20Via%20Spatial%20Beam%20Search%20in%20Coarse-To-Fine%20Reinforcement%20Learning%22) — *Mon 10:21, Room 411/412*  
  Coarse-to-fine value-based RL with beam search over discretization tree and perceptual gating; +12% on RLBench, minutes of real training.
- **Climb with SHERPA: Heuristic-Guided Reinforcement Learning Via Segmented Experience Relay** — Kim, Lee, Lee et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Climb%20with%20SHERPA%3A%20Heuristic-Guided%20Reinforcement%20Learning%20Via%20Segmented%20Experience%20Relay%22) — *Wed 15:38, Room 329*  
  Alternates heuristic and RL control in contiguous segments to guide exploration in sparse-reward long-horizon manipulation.
- ✅ **E2HiL: Entropy-Guided Sample Selection for Efficient Real-World Human-In-The-Loop Reinforcement Learning** — Deng, Lin, XUE et al. — [arXiv:2601.19969](https://arxiv.org/abs/2601.19969) — *Mon 10:27, Room 411/412*  
  Selects human-in-the-loop RL samples via influence on policy entropy, improving success 25% with fewer interventions on real robots.
- ✅ **FastDSAC: Enhancing Policy Plasticity Via Constrained Exploration for Scalable Humanoid Locomotion** — Lu, Dun, zhou et al. — [arXiv:2606.31691](https://arxiv.org/abs/2606.31691) — *Mon 9:00, Room 315/316*  
  Truncated-Gaussian distributional SAC for massively parallel high-UTD training preserves plasticity and speeds humanoid locomotion learning.
- ✅ **CDE: Concept-Driven Exploration for Reinforcement Learning** — Mao, Liu, Zabounidis et al. — [arXiv:2510.08851](https://arxiv.org/abs/2510.08851) — *Tue 15:25, Room 409/410*  
  VLM-generated object concepts reconstructed via an auxiliary loss provide intrinsic rewards for targeted exploration in visual RL.
- ✅ **VE2VF: Vision-Enabled to Vision-Free Distillation Via Real-World Reinforcement Learning for Robust Contact-Rich Manipulation** — Kowalski, Li, Lee — [arXiv:2605.29564](https://arxiv.org/abs/2605.29564) — *Mon 10:12, Room 411/412*  
  Real-world human-in-the-loop RL with vision-enabled teacher distilled to vision-free student generalizes across NIST assembly variants.
- ✅ **Where-To-Learn: Analytical Policy Gradient Directed Exploration for On-Policy Robotic Reinforcement Learning** — Chang, Yao, Liu et al. — [arXiv:2603.27317](https://arxiv.org/abs/2603.27317) — *Mon 15:38, Room 320*  
  Directed exploration for PPO using analytical policy gradients from differentiable dynamics to steer toward high-reward regions.

### Continual Learning, Adaptation & Cross-Embodiment
- ✅ **LIDEA: Human-To-Robot Imitation Learning Via Implicit Feature Distillation and Explicit Geometry Alignment** — Xu, Lin, Zhan et al. — [arXiv:2604.10677](https://arxiv.org/abs/2604.10677) — *Tue 15:08, Room 317/318*  
  Learns policies from human videos via latent distillation aligning human/robot features and embodiment-agnostic 3D geometry; replaces 80% robot demos.
- **Collective Learning for Unified Robotic Manipulation Via Predictive Dynamics Adaption** — Liu, Meng, Bing et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Collective%20Learning%20for%20Unified%20Robotic%20Manipulation%20Via%20Predictive%20Dynamics%20Adaption%22) — *Mon 14:45, Room 302/303*  
  Separates shared task intent from embodiment dynamics via a predictive-dynamics latent regularizer, avoiding collapse when pooling multi-robot data.
- **Intra-Rollout Self-Supervised Adaptation of Inverse Dynamics Models for Sub-Centimeter Precision Placement** — Xu, Sun, Yi et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Intra-Rollout%20Self-Supervised%20Adaptation%20of%20Inverse%20Dynamics%20Models%20for%20Sub-Centimeter%20Precision%20Placement%22) — *Wed 9:13, Room 401/402*  
  Adapts inverse dynamics models online within a rollout using self-generated transitions plus decaying stochastic excitation to reach sub-centimeter placement.
- ✅ **RoboHarness: A Memory-Augmented Policy Harness for Vision-Language-Action Model Robustness Via In-Context Adaptation** — Li, Li, Zhou et al. — [arXiv:2603.24060](https://arxiv.org/abs/2603.24060) — *Wed 9:32, Room 403/404*  
  Memory-augmented harness with retrieval and failure attribution lets frozen VLAs adapt in context to OOD perturbations.
- **Relational Dexterity: Transferring Human Dexterity Via Relational Geometric Prior** — Chen, Zhang, Liu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Relational%20Dexterity%3A%20Transferring%20Human%20Dexterity%20Via%20Relational%20Geometric%20Prior%22) — *Wed 9:55, Room 304/305*  
  Embodiment-invariant relational geometric prior from human hand data guides curriculum RL for dexterous transfer.
- ✅ **CEI: A Unified Interface for Cross-Embodiment Visuomotor Policy Learning in 3D Space** — Wu, Li, Gong et al. — [arXiv:2601.09163](https://arxiv.org/abs/2601.09163) · [RA-L](https://doi.org/10.1109/lra.2026.3656802) — *Mon 14:48, Room 302/303*  
  Cross-embodiment interface aligning trajectories by functional similarity to transfer demos across arms and end-effectors.

### Humanoid & Legged Locomotion via RL
- ✅ **APEX: Action Priors Enable Efficient Exploration for Robust Motion Tracking on Legged Robots** — Sood, Nakhwa, SUN et al. — [arXiv:2511.09091](https://arxiv.org/abs/2511.09091) — *Wed 14:58, Room 304/305*  
  Decaying action priors from demonstrations guide early RL exploration for motion tracking, yielding pure RL policy without reference inputs.
- **Role-Based Reward Decomposition for Legged Locomotion Reinforcement Learning** — Tian, Trahanias — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Role-Based%20Reward%20Decomposition%20for%20Legged%20Locomotion%20Reinforcement%20Learning%22) — *Mon 15:27, Room 320*  
  Careful study of stance/swing role-based reward decomposition with multi-critic variants; shows when decomposition helps or hurts under objective conflict.
- ✅ **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation** (Award Candidate) — He, Xu, Li et al. — [arXiv:2603.03279](https://arxiv.org/abs/2603.03279) — *Mon 9:11, Room 406*  
  Physics-based retargeting, distilled latent skill controller and RL fine-tuning enable goal-conditioned humanoid loco-manipulation from egocentric perception.
- **Learning Humanoid Agile and Contact-Rich Control from Latent Predictive Motion Priors** — He, Dong, Zhang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Learning%20Humanoid%20Agile%20and%20Contact-Rich%20Control%20from%20Latent%20Predictive%20Motion%20Priors%22) — *Mon 9:06, Room 315/316*  
  Latent predictive motion prior from task-agnostic human motion replaces heavy reward shaping for agile humanoid RL.
- ✅ **DreamControl-V2: Simpler and Scalable Autonomous Humanoid Skills Via Trainable Guided Diffusion Priors** — Harithas, Kwak, Katara et al. — [arXiv:2604.00202](https://arxiv.org/abs/2604.00202) — *Tue 9:46, Room 320*  
  Trains guided diffusion motion priors directly in robot space to guide RL for humanoid loco-manipulation skills.
- ✅ **GMT: General Motion Tracking for Humanoid Whole-Body Control** — Chen, Ji, Cheng et al. — [arXiv:2506.14770](https://arxiv.org/abs/2506.14770) — *Tue 10:08, Room 320*  
  Single unified humanoid motion-tracking policy using adaptive motion sampling and motion MoE for real-world whole-body control.

### Data Generation & Collection
- ✅ **Scalable Multi-Task Data Generation Via Reinforcement Learning for Language-Conditioned Bimanual Dexterous Manipulation** — Li, Jin, Liu et al. — [arXiv:2606.22471](https://arxiv.org/abs/2606.22471) — *Tue 10:24, Room 302/303*  
  RL-based pipeline with generalizable rewards and domain randomization to synthesize language-conditioned bimanual dexterous datasets.
- ✅ **RADAR: Closed-Loop Robotic Data Generation Via Semantic Planning and Autonomous Causal Environment Reset** — Wang, Zhu, Zhong et al. — [arXiv:2603.11811](https://arxiv.org/abs/2603.11811) — *Tue 9:00, Room 302/303*  
  Fully autonomous closed-loop data engine with VLM task planning and causal environment resets from 2-5 demonstrations.
- ✅ **CRAFT: Video Diffusion for Bimanual Robot Data Generation** — Chen, Liu, Sukhatme et al. — [arXiv:2604.03552](https://arxiv.org/abs/2604.03552) — *Tue 14:36, Room 317/318*  
  Uses Canny-conditioned video diffusion to turn sim trajectories into diverse photorealistic bimanual demos with action labels.

### Additional
- **CRISP: Context-Robust Skill Inference for Long-Horizon Offline Robotic Control** — Zaidi, Munir, Zaidi — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22CRISP%3A%20Context-Robust%20Skill%20Inference%20for%20Long-Horizon%20Offline%20Robotic%20Control%22) — *Mon 9:26, Room 409/410*  
  Hierarchical offline RL with masked latent skill inference, diffusion skill execution and conservative latent values; robust to missing context.
- ✅ **Learning Hierarchical Skill Policies with Offline Quality-Diversity Reinforcement Learning** — Anakewat, Osa, Harada — [arXiv:2608.19684](https://arxiv.org/abs/2608.19684) — *Mon 10:19, Room 320*  
  Advantage-weighted quality-diversity offline skill extraction plus dataset reuse for offline-to-online hierarchical RL in sparse-reward tasks.
- ✅ **VLASH: Real-Time VLAs Via Future-State-Aware Asynchronous Inference** — Tang, Sun, Zhao et al. — [arXiv:2512.01031](https://arxiv.org/abs/2512.01031) — *Wed 9:26, Room 403/404*  
  Future-state-aware asynchronous inference lets VLAs act continuously without stalls, improving reaction latency on dynamic tasks.
- ✅ **IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies** — Hosseinkhani, Peng, Shramko et al. — [arXiv:2609.10915](https://arxiv.org/abs/2609.10915) — *Mon 9:00, Room 329*  
  Replaces iterative diffusion/flow action heads with a single-step cIMLE generator for fast VLA action generation.

## P2 — Nice to Read

### VLA Models
- **TempoFit: Plug-And-Play Layer-Wise Temporal KV Memory for Long-Horizon VLA Manipulation** — Sun, Yang, Zhang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22TempoFit%3A%20Plug-And-Play%20Layer-Wise%20Temporal%20KV%20Memory%20for%20Long-Horizon%20VLA%20Manipulation%22) — *Wed 15:42, Room 403/404*  
  Training-free temporal memory for frozen VLAs by reusing layer-wise prefix K/V caches across timesteps with recency bias.
- **Embodied Chain-Of-Thought Model Via Interleaved Reasoning** — Liu, Zhao, Zhang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Embodied%20Chain-Of-Thought%20Model%20Via%20Interleaved%20Reasoning%22) — *Wed 15:54, Room 409/410*  
  Embodied CoT VLA that interleaves subgoal image prediction and textual reasoning in one pass using modality-specialized MoE experts.
- ✅ **Improving Vision-Language-Action Model Fine-Tuning with Structured Stage and Keyframe Supervision** — Xu, Chen, Wang et al. — [arXiv:2606.26801](https://arxiv.org/abs/2606.26801) — *Wed 14:43, Room 411/412*  
  Auxiliary stage-classification and next-gripper-keyframe heads derived automatically from demos improve VLA fine-tuning on long-horizon tasks.
- ✅ **M²-VLA: Boosting Vision-Language Models for Generalizable Manipulation Via Layer Mixture and Meta-Skills** — Xiao, Zhang, Liu et al. — [arXiv:2604.24182](https://arxiv.org/abs/2604.24182) — *Wed 14:34, Room 411/412*  
  Keeps a generalist VLM backbone with mixture-of-layers feature extraction and a meta-skill module to avoid forgetting from end-to-end VLA fine-tuning.
- **DexTact: A Visuo-Tactile Foundation Model for Dexterous Hand Grasping** — Fang, Tao, Li — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22DexTact%3A%20A%20Visuo-Tactile%20Foundation%20Model%20for%20Dexterous%20Hand%20Grasping%22) — *Tue 14:52, Room 328*  
  Visuo-tactile VLA for dexterous grasping; ablations show tactile closes lighting gaps and success follows power-law data scaling.
- ✅ **3D HAMSTER: Bridging Planning and Control in Hierarchical Vision Language Action Models through 3D Trajectory Guidance** — Hwang, Lee, Kim et al. — [arXiv:2606.31329](https://arxiv.org/abs/2606.31329) — *Tue 15:22, Room 317/318*  
  Hierarchical VLA where the VLM predicts 3D rather than 2D end-effector trajectories to guide point-cloud low-level policies.
- **Hand-Centric Interaction Representation for Vision-Language-Action Models** — Kaichi, Kambara, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Hand-Centric%20Interaction%20Representation%20for%20Vision-Language-Action%20Models%22) — *Wed 14:49, Room 411/412*  
  Hand-centric flow-matching action representation from human video (pose plus contact) for VLA pretraining.
- ✅ **CorridorVLA: Explicit Spatial Constraints for Generative Action Heads Via Sparse Anchors** — Li, Chen, Zhang et al. — [arXiv:2604.21241](https://arxiv.org/abs/2604.21241) — *Tue 15:16, Room 317/318*  
  Predicts sparse spatial anchors defining a tolerance corridor that constrains the flow-matching action head of VLAs; gains on LIBERO-Plus.
- ✅ **Feeling the Unexpected: ResTacVLA for Contact-Rich Manipulation Via Residual Tactile Representation** — Zhang, Xie, Meng et al. — [arXiv:2607.03387](https://arxiv.org/abs/2607.03387) — *Mon 15:57, Room 302/303*  
  Feeds VLAs residual tactile signals (discrepancy from visual prediction) via VQ primitives, gated by visual uncertainty, to avoid modality collapse.
- ✅ **DualCoT-VLA: Visual-Linguistic Chain of Thought Via Parallel Reasoning for Vision-Language-Action Models** — Zhong, Li, He et al. — [arXiv:2603.22280](https://arxiv.org/abs/2603.22280) — *Tue 14:30, Room 317/318*  
  VLA with parallel visual and linguistic chain-of-thought via learnable queries, avoiding autoregressive reasoning latency.
- ✅ **OG-VLA: Orthographic Image Generation for 3D-Aware Vision-Language Action Model** — Singh, Goyal, Birchfield et al. — [arXiv:2506.01196](https://arxiv.org/abs/2506.01196) — *Mon 10:18, Room 401/402*  
  3D-aware VLA rendering canonical orthographic views and generating end-effector target images; strong generalization on Arnold/Colosseum.
- ✅ **Point What You Mean: Visually Grounded Instruction Policy** — Yu, Zhao, Liu et al. — [arXiv:2512.18933](https://arxiv.org/abs/2512.18933) — *Wed 15:10, Room 411/412*  
  Augments VLA instructions with visual pointing cues (bounding boxes) and auto-annotation to resolve referring ambiguity.

### Robot Learning & Imitation
- **Energy-Score Stabilized Attentive Mixture-Density Policy: A Lightweight Approach to Multimodal Robot Motion** — Fujita, Ichiwara, Sugano et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Energy-Score%20Stabilized%20Attentive%20Mixture-Density%20Policy%3A%20A%20Lightweight%20Approach%20to%20Multimodal%20Robot%20Motion%22) — *Tue 14:30, Room 320*  
  Mixture-density policy trained with Energy Score instead of NLL gives single-pass multimodal actions, beating Diffusion Policy in small-data regimes.
- ✅ **VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning** — Zhou, Xue, Ye et al. — [arXiv:2603.17720](https://arxiv.org/abs/2603.17720) — *Wed 15:38, Room 328*  
  Lifts image features into a volumetric representation with learned voxel token selection for imitation policies, large gains on LIBERO and ManiSkill.
- ✅ **SSI-Policy: Learning Structured Scene Interfaces for Vision-Language Robotic Manipulation** — Wang, Ouyang, Wu et al. — [arXiv:2606.26800](https://arxiv.org/abs/2606.26800) — *Tue 15:25, Room 317/318*  
  RGB-only structured scene interface (depth, layouts, 2D motion) trainable from action-free video enables few-demo policies and cross-embodiment.
- ✅ **CubeDAgger: Interactive Imitation Learning for Dynamic Systems with Efficient yet Low-Risk Interaction** — Kobayashi — [arXiv:2505.04897](https://arxiv.org/abs/2505.04897) — *Tue 10:09, Room 403/404*  
  Interactive imitation learning for dynamic tasks using consensus of action candidates and colored-noise exploration to keep stability.
- ✅ **LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning** — Rodriguez Jimenez, Li, Mazza et al. — [arXiv:2603.08476](https://arxiv.org/abs/2603.08476) — *Mon 9:41, Room 409/410*  
  Mixture-of-experts imitation with routing regularized by an unsupervised learned latent skill space, avoiding phase labels.

### RL Algorithms & Exploration
- ✅ **LEACL: LLM-Enhanced Automatic Curriculum Learning for Reinforcement Learning in Long-Horizon Manipulation Tasks** — Heravi, Ouyang, Xu et al. — [arXiv:2607.23515](https://arxiv.org/abs/2607.23515) — *Tue 15:22, Room 409/410*  
  Uses LLMs to automatically construct curricula for RL on long-horizon sparse-reward manipulation.
- ✅ **MO-Playground: Massively Parallelized Multi-Objective Reinforcement Learning for Robotics** — Janwani, Novoseller, Lawhern et al. — [arXiv:2603.09237](https://arxiv.org/abs/2603.09237) · [RA-L](https://doi.org/10.1109/lra.2026.3700381) — *Tue 14:48, Room 409/410*  
  GPU-native multi-objective RL algorithm and parallel environment suite for Pareto policy families in robotics.
- ✅ **Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning** — Shin, Ren, Xiong et al. — [arXiv:2603.19078](https://arxiv.org/abs/2603.19078) — *Wed 15:57, Room 409/410*  
  Policy network architecture built on Articulated Body Algorithm inertia propagation improves RL sample efficiency and dynamics-shift robustness.
- ✅ **WARL: Wrench-Augmented Reinforcement Learning for Task-Agnostic Learning in Legged Robots** — Yoneda, Kawaharazuka, Okada — [arXiv:2607.24036](https://arxiv.org/abs/2607.24036) — *Wed 15:07, Room 304/305*  
  Adds wrench actions for exploration in legged RL, annealed by curriculum; analyzes when it hurts embodiment use.

### Offline RL & Skill Learning
- **Robotic Long-Horizon Manipulation with Bayesian Non-Parametric Skill Priors** — Meng, Yao, Wu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Robotic%20Long-Horizon%20Manipulation%20with%20Bayesian%20Non-Parametric%20Skill%20Priors%22) — *Mon 10:09, Room 411/412*  
  Dirichlet-process mixture skill prior for hierarchical RL improves exploration on sparse-reward long-horizon manipulation vs. single-Gaussian priors.
- ✅ **ARP: Enhancing Quantized Skill Abstractions via Visual Alignment and Iterative Refinement for Robotic Manipulation** — Wang, Jia, Duan et al. — [arXiv:2606.22480](https://arxiv.org/abs/2606.22480) — *Mon 14:53, Room 302/303*  
  Discrete skill imitation with contrastive visual-action alignment and an iterative residual head to fix quantization error.
- ✅ **Learning Semantic Atomic Skills for Multi-Task Robotic Manipulation** — Zhu, Wang, Wu et al. — [arXiv:2512.18368](https://arxiv.org/abs/2512.18368) — *Mon 14:56, Room 302/303*  
  Learns semantically aligned atomic skill space from demos via contrastive alignment with keypose imagination for skill chaining.

### Humanoid & Legged Locomotion via RL
- ✅ **LooperMuscle: Fast and Stable Learning of Humanoid Whole-Body Tracking Via Structured Mixture-Of-Experts** — Liu, Li, Yu et al. — [arXiv:2608.00820](https://arxiv.org/abs/2608.00820) — *Mon 15:08, Room 401/402*  
  Structured mixture-of-experts actor, distributional critic and routed replay close gap between FastSAC and PPO for humanoid tracking in 45 min.
- ✅ **APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots** — Wang, Leng, Lin et al. — [arXiv:2602.11143](https://arxiv.org/abs/2602.11143) — *Mon 14:34, Room 401/402*  
  Humanoid climbing via a ratchet best-so-far progress reward for safe goal-reaching exploration, distilled into one multi-skill policy on Unitree G1.
- ✅ **GeCCo - a Generalist Contact-Conditioned Policy for Loco-Manipulation Skills on Legged Robots** — Atanassov, Yu, Gangapurwala et al. — [arXiv:2509.17582](https://arxiv.org/abs/2509.17582) — *Tue 9:33, Room 320*  
  Single RL contact-conditioned tracking policy serves as a planner-agnostic interface for diverse quadruped loco-manipulation skills.

### Efficient VLA Inference
- **TIDAL: Temporally Interleaved Diffusion and Action Loop for Dynamic Manipulation** — Sun, Wang, Bai et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22TIDAL%3A%20Temporally%20Interleaved%20Diffusion%20and%20Action%20Loop%20for%20Dynamic%20Manipulation%22) — *Wed 15:16, Room 411/412*  
  Dual-frequency scheduler interleaving single-step flow integration with execution and staleness-aware training for dynamic VLA manipulation.
- ✅ **Shallow-π: Knowledge Distillation for Flow-Based VLAs** (Award Candidate) — Jeon, Choi, Kim — [arXiv:2601.20262](https://arxiv.org/abs/2601.20262) — *Mon 14:30, Room 406*  
  Distills flow-based VLA (pi0-style) from 18 to 6 transformer layers, 2x faster inference with <1% success drop on Jetson.
- ✅ **DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference** — Li, Ma, Ding et al. — [arXiv:2603.10469](https://arxiv.org/abs/2603.10469) — *Wed 14:55, Room 411/412*  
  Training-free depth-guided visual token merging speeds VLA inference 1.28x with under 1% success loss across three VLAs.
- ✅ **Fast Enough to Act: Spatio-Temporal Visual Token Merging for Low-Latency Robotic VLMs and VLAs** — Chen, Wang, Zhou — [arXiv:2606.29350](https://arxiv.org/abs/2606.29350) — *Mon 9:55, Room 317/318*  
  Training-free spatiotemporal visual token merging with positional correction gives up to 8.3x speedup on pi0.5 at high resolution.

### Additional
- **StageNav: A Diagnostic Benchmark and State-Aligned RL Framework for Long-Horizon Navigation** — Fang, Yang, Liu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22StageNav%3A%20A%20Diagnostic%20Benchmark%20and%20State-Aligned%20RL%20Framework%20for%20Long-Horizon%20Navigation%22) — *Wed 10:09, Room 409/410*  
  Reinforcement fine-tuning of LVLM navigators with dense process rewards from verified subgoal-state predictions, plus diagnostic long-horizon benchmark.
- **JOP-VLN: Joint On-and-Off Policy Learning for Vision-and-Language Navigation** — He, Zhao, Zheng et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22JOP-VLN%3A%20Joint%20On-and-Off%20Policy%20Learning%20for%20Vision-and-Language%20Navigation%22) — *Wed 15:11, Room 401/402*  
  Three-stage VLN training combining IL, DAgger and joint on/off-policy RL with high-entropy sampling, SOTA on R2R.
- **Safe Offline Reinforcement Learning via Chance-Constrained Policy Filtering** — Kim, Yoo, Ahn — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Safe%20Offline%20Reinforcement%20Learning%20via%20Chance-Constrained%20Policy%20Filtering%22) — *Mon 9:00, Room 320*  
  Chance-constrained policy filter with posterior-calibrated cost estimator for safe offline RL with adjustable thresholds at deployment.
- ✅ **Directional Constraints for Efficient Exploration in Safe Reinforcement Learning** — Magliano, Liu, Peters et al. — [arXiv:2607.12784](https://arxiv.org/abs/2607.12784) — *Tue 15:16, Room 320*  
  Extends the ATACOM safety layer with directional constraints that only activate when moving toward constraint boundaries, improving safety-performance trade-off.
- ✅ **NavThinker: Action-Conditioned World Model for Coupled Prediction and Planning in Social Navigation** — HU, Gong, Kong et al. — [arXiv:2603.15359](https://arxiv.org/abs/2603.15359) — *Mon 14:33, Room 403/404*  
  Action-conditioned world model in Depth-Anything feature space feeds think-ahead features and reward shaping to PPO for social navigation.
- **Physics-Informed Policy Learning for Floating-Base Robots** — Yang, Liu, Ding et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Physics-Informed%20Policy%20Learning%20for%20Floating-Base%20Robots%22) — *Wed 9:40, Room 304/305*  
  Grey-box Lagrangian dynamics model with learned contact forces as a drop-in for MBPO, yielding more reliable imagined rollouts for locomotion.
- ✅ **Demonstration-Free Robotic Control Via LLM Agents** — Tsui, Fang, Hwu — [arXiv:2601.20334](https://arxiv.org/abs/2601.20334) — *Tue 9:33, Room 328*  
  Runs an unmodified frontier LLM coding agent as a demonstration-free manipulation controller, approaching few-shot VLA success on LIBERO/ManiSkill/MetaWorld.
- ✅ **StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models** — Pangaonkar, Rath, Patil et al. — [arXiv:2603.20659](https://arxiv.org/abs/2603.20659) — *Mon 15:27, Room 334*  
  Training-free VLM reasoning over rollout videos to rearrange initial scene state, mitigating VLA distractor and obstruction failures.
- ✅ **SPIDER: Scalable Physics-Informed Dexterous Retargeting** — Pan, Wang, Qi et al. — [arXiv:2511.09484](https://arxiv.org/abs/2511.09484) — *Tue 10:27, Room 302/303*  
  Physics-based retargeting converts kinematic human demonstrations into dynamically feasible robot data for dexterous and humanoid policy learning.
- ✅ **Kinematics-Aware Diffusion Policy with Consistent 3D Observation and Action Space for Whole-Arm Robotic Manipulation** — Lv, Yu, Jia et al. — [arXiv:2512.17568](https://arxiv.org/abs/2512.17568) — *Mon 14:42, Room 302/303*  
  Represents whole-arm states and actions as 3D body points aligned with point clouds, with kinematic priors in diffusion, improving spatial generalization.
- **RobustFuzz: A Semantic Disturbance Benchmark for Learned Policies in Embodied Control** — Ren — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22RobustFuzz%3A%20A%20Semantic%20Disturbance%20Benchmark%20for%20Learned%20Policies%20in%20Embodied%20Control%22) — *Mon 10:04, Room 411/412*  
  Fuzzing benchmark that searches for timing/hardware disturbance schedules that break learned control policies.
- **What Good Is a Good Robot Model? Differentiable Regression, Parameter Ablation, and Policy Performance of Quadrupedal Models** — Hackett, Hubicki — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22What%20Good%20Is%20a%20Good%20Robot%20Model%3F%20Differentiable%20Regression%2C%20Parameter%20Ablation%2C%20and%20Policy%20Performance%20of%20Quadrupedal%20Models%22) — *Wed 15:51, Room 304/305*  
  Systematically compares fine-tuned parametric models vs domain randomization for quadruped sim-to-real policy performance.
- ✅ **CORAL: Scalable Multi-Task Robot Learning Via LoRA Experts** — Luo, Chen, Liang et al. — [arXiv:2603.09298](https://arxiv.org/abs/2603.09298) — *Mon 9:15, Room 411/412*  
  Frozen VLA backbone with swappable per-task LoRA experts avoids negative transfer in multi-task deployment.
- ✅ **VR-DAgger: Immersive VR for Dexterous Data Collection and Uncertainty-Guided On-Policy Correction** — Zurbrügg, Portela, Bhardwaj et al. — [arXiv:2605.27114](https://arxiv.org/abs/2605.27114) — *Wed 9:04, Room 302/303*  
  VR teleoperation with MC-dropout uncertainty selecting failure segments for targeted DAgger-style corrections of dexterous diffusion policies.
- **DiCT: Disentangled Control for Safe and Generalizable Robotic Manipulation with Diffusion Policies** — Xu, Jingwei, Yang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22DiCT%3A%20Disentangled%20Control%20for%20Safe%20and%20Generalizable%20Robotic%20Manipulation%20with%20Diffusion%20Policies%22) — *Wed 9:45, Room 329*  
  Modular disentangled control adds obstacle-awareness to diffusion policies without retraining the manipulation skill.

## Workshops

IROS 2026 has 86 workshops and tutorials, on Sunday Sep 27 and Thursday Oct 1. These are the ones closest to robot learning, VLAs, world models, and sim-to-real.

**Thursday, Oct 1:**
- **Physical World Models for Scaling Embodied AI** — afternoon, Rooms 319 & 320 — Haibao Yu, Dandan Zhang, Lei Yang, et al. — [workshop page](https://physical-world-models.github.io/IROS2026/)
- **Building Scalable Infrastructure for Robot Learning: From Data Scaling to Real-World Deployment (ScaleInfra)** — morning, Rooms 317 & 318 — Chao Yu, Yu Wang, Huazhe Xu, Shenyuan Gao, Zhongyu Li, Koushil Sreenath — [workshop page](https://scale-infra.github.io/iros2026/)
- **Sim2Real and Classical Control: From Rigorous Theory to Data-Driven Robotics** — full day, Rooms 303 & 304 — Dario Sanalitro, Enrico Ferrentino, Silvia Tulli, Sai Kishor Kothakota, Zhixuan Liu, Jonathan Francis — [workshop page](https://sim2realgap.github.io/sim2real-and-control-workshop-iros2026/)
- **3rd Workshop on AI Meets Autonomy: Vision, Language, and Autonomous Systems** — morning, Rooms 319 & 320 — Wenshan Wang, Ji Zhang, et al. — [workshop page](https://www.ai-meets-autonomy.com/)
- **Reproducible Benchmarking of Robotic Grasping and Manipulation: From Advanced AI to Generalized Humanoid Intelligence** — afternoon, Rooms 315 & 316 — [workshop list](https://2026.ieee-iros.org/program/workshops/)

**Sunday, Sep 27** (already happened, but worth checking the accepted papers and talks):
- **WORLDS: World Models and Spatial Intelligence for Physical AI** — [workshop page](https://worlds-iros2026.github.io/)
- **Bridging the Gap between Neural and Symbolic World Models for Robot Planning, Reasoning, and Action (RoBoWoMo)** — [workshop page](https://worldmodelworkshop.github.io/)
- **Search Algorithms for Robot Learning** — [workshop page](https://sites.google.com/view/search-for-robot-learning-2026/)
- **Compositional and Modular Learning in the Era of Scaling in Robotics** — [workshop page](https://compositional-robotics.github.io/)
- **1st International Workshop on Industrial Applications of Robot Learning (IARL2026)** — [workshop page](https://aistairc.github.io/IROS2026-workshop/)
- **Scaling vs. Structure: Rethinking Bimanual Manipulation Beyond Single-Arm Policies** — [workshop page](https://bimanual-robot-learning.github.io/)

How the workshops connect to the papers:
- Physical World Models, WORLDS and RoBoWoMo go with the World Models sections above.
- ScaleInfra connects to the RL fine-tuning and data-generation papers.
- Sim2Real & Classical Control pairs with the generalization and sim-to-real papers.
- (See the [full workshop list](https://2026.ieee-iros.org/program/workshops/) for everything else on offer.)

## Themes Worth Following

A few threads run through this year's program:
- **RL fine-tuning of VLAs.** This is the robotics side of the RLVR/GRPO work on LLMs. Start with VLA-RL, AtomVLA, and Foresight Residual RL, and see TD-GRPC and Beyond Imitation for GRPO-style updates.
- **Careful empirical studies of imitation learning.** Geometric Entropy, Learning from the Best, TRACE, MaskVLA, and Beyond Implicit Force each isolate one factor behind why imitation succeeds or fails.
- **Evaluating how well VLAs generalize.** Start with How VLAs (Really) Work, REALM, LangGap, and The Moving Eye.
- **World models.** Here they are used for policy learning, data synthesis, and evaluation. GrndCtrl post-trains a world model with RLVR-style verifiable rewards.

## How I Built This List

The IROS program site publishes an abstract for almost every paper. I used those rather than titles alone.
- I started from all 1,933 papers.
- I kept the 747 whose title or abstract mentions a core learning topic.
- I scored each of those against my group's research areas using the full abstract, then checked the result by hand.

arXiv preprints (✅) were matched by title where available. Some papers without a preprint may have one posted after the conference. If you're at IROS 2026 in Pittsburgh and this overlaps with your interests, I hope it saves you some filtering time.
