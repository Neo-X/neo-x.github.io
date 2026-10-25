---
title: "My IROS 2026 Reading List: Robot Learning, VLAs, RL Fine-Tuning & World Models"
date: 2026-09-28
description: "A prioritized reading list of 142 IROS 2026 papers on robot learning, VLA models, RL fine-tuning of robot policies, imitation learning, diffusion and flow policies, world models, and generalization, filtered from the 1,929 papers in the official program using their abstracts, plus the relevant workshops."
summary: "I filtered all 1,929 IROS 2026 papers down to a prioritized list for my own research areas — robot learning, VLA models, RL fine-tuning of robot policies, imitation learning, diffusion/flow policies, world models, and generalization — plus the related workshops. Sharing it here in case it's useful to others in Pittsburgh."
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
cover: <img width="100%" src="/assets/projects/IROS2026-reading-list.svg" alt="IROS 2026 reading list, 1,929 papers filtered to 142">
---

# My IROS 2026 Reading List

<div align="center"><img src="/assets/projects/IROS2026-reading-list.svg" alt="Funnel from the 1,929 papers in the IROS 2026 program to 742 candidates on the core topics to 142 on the reading list: 38 must read, 74 should read, and 30 nice to read" width="80%"></div>

IROS 2026 in Pittsburgh has 1,929 papers in its program. Nobody is reading all of them, so I filtered the list down to what's relevant to my group's research: **robot learning, vision-language-action (VLA) models, RL fine-tuning of robot policies, imitation learning, diffusion and flow policies, world models, and generalization.** I'm sharing the filtered list here in case it's useful to others at the conference.

Papers are grouped into **P0 (must read, 38 papers)**, **P1 (should read, 74)**, and **P2 (nice to read, 30)**. Each entry has a one-line summary written from the abstract, the time and room of its talk, and its poster: the paper number (e.g. **#3006**, which is how the official paper index and poster map identify each poster), its session code, and the board and face (e.g. *B4, North face*) in Exhibit Hall B, where every poster session is held.
- ✅ means you can read the paper now: it has an arXiv preprint, which I've linked directly, or it is a published RA-L paper.
- RA-L papers presented at IROS link to IEEE Xplore.
- The IROS 2026 proceedings aren't on IEEE Xplore yet. For the remaining papers, the IEEE Xplore link is a title search that will find the paper once the proceedings are posted.

**Download the list as BibTeX** for Paperpile, Zotero, Mendeley, or any reference manager: [all 142 papers](/assets/bib/IROS_2026_reading_list.bib) · [P0 must read](/assets/bib/IROS_2026_reading_list_P0.bib) · [P1 should read](/assets/bib/IROS_2026_reading_list_P1.bib) · [P2 nice to read](/assets/bib/IROS_2026_reading_list_P2.bib). Each entry includes the abstract, the arXiv ID where there is one, and the talk and poster times and locations.

<!--more-->

## P0 — Must Read

### VLA Models
- ✅ **Recursive Belief Vision Language Action Models** — Bagaria, Patel, Sebastian — [arXiv:2602.20659](https://arxiv.org/abs/2602.20659) — *Talk Tue 15:45, Room 317/318 · **Poster #436** (session T2LBY04): board F10 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  VLA with a recursive latent belief trained via world-model objectives for partial observability; big long-horizon gains over pi0.
- ✅ **Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos** — XU, Zhang, Wang et al. — [arXiv:2606.18955](https://arxiv.org/abs/2606.18955) — *Talk Tue 14:33, Room 317/318 · **Poster #452** (session T2LAX04): board D9 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  Disentangled VQ-VAE latent actions extract motion priors from unlabeled human egocentric video for cross-embodiment VLA pretraining.
- ✅ **Enhancing Generalization in Vision–Language–Action Models by Preserving Pretrained Representations** — Grover, Gopalkrishnan, Ai et al. — [arXiv:2509.11417](https://arxiv.org/abs/2509.11417) — *Talk Wed 14:55, Room 411/412 · **Poster #978** (session W2LAY08): board E10 (West face), Exhibit Hall B, Wed 16:30–18:00*  
  Preserves VLM representations in VLA fine-tuning via frozen+trainable dual encoders, string action tokens, and VL co-training, improving generalization.
- ✅ **EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos** — Yang, Yu, Wu et al. — [arXiv:2507.12440](https://arxiv.org/abs/2507.12440) — *Talk Tue 15:19, Room 317/318 · **Poster #1502** (session T2LBX04): board F9 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  Trains a VLA on egocentric human video predicting wrist/hand actions, retargets to robots and fine-tunes with few demos, ablating human data value.
- ✅ **Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions** — Feng, Jain — [arXiv:2608.03563](https://arxiv.org/abs/2608.03563) — *Talk Tue 15:48, Room 317/318 · **Poster #2617** (session T2LBY04): board F8 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Changes the VLA prediction target to a unified latent encoding motor control and visual scene transitions, improving training efficiency without architecture changes.
- ✅ **MaskVLA: Visual Masking against Trajectory Overfitting of Vision-Language-Action Model** — Jiang, Huang, Wang et al. — [arXiv:2609.23565](https://arxiv.org/abs/2609.23565) — *Talk Mon 10:01, Room 411/412 · **Poster #1686** (session M1LBX08): board E11 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Finds VLAs severely overfit trajectories when fine-tuned on small data; random main-camera masking forces use of wrist-view features.

### RL Fine-Tuning & Policy Optimization
- ✅ **AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models** — Sun, XU, Cao et al. — [arXiv:2603.08519](https://arxiv.org/abs/2603.08519) — *Talk Tue 10:09, Room 302/303 · **Poster #33** (session T1LBY01): board B3 (North face), Exhibit Hall B, Tue 11:00–12:30*  
  Decomposes demos into LLM-generated subtasks and uses a latent world model to score action chunks, enabling offline GRPO post-training of VLAs.
- ✅ **Beyond Imitation: Reinforcement Learning Fine-Tuning for Adaptive Diffusion Navigation Policies** — Sheng, Bai, Xu et al. — [arXiv:2603.12868](https://arxiv.org/abs/2603.12868) — *Talk Wed 9:06, Room 409/410 · **Poster #1850** (session W1LAX07): board C13 (West face), Exhibit Hall B, Wed 11:00–12:30*  
  Fine-tunes diffusion navigation policies with GRPO over sampled trajectories, avoiding value networks; improves OOD safety and transfers to real.
- ✅ **Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models** — Liu, Zhang, Liu et al. — [arXiv:2607.16506](https://arxiv.org/abs/2607.16506) — *Talk Mon 9:16, Room 409/410 · **Poster #4421** (session M1LAX07): board F9 (West face), Exhibit Hall B, Mon 11:00–12:30*  
  Residual RL over frozen VLA with learned foresight value rewards shapes subtask handoff states, raising chained-task success from 54% to 86%.
- ✅ **Human-In-The-Loop Online Rejection Sampling for Robotic Manipulation** — Lu, Zhao, Lin et al. — [arXiv:2510.26406](https://arxiv.org/abs/2510.26406) — *Talk Mon 10:24, Room 411/412 · **Poster #4652** (session M1LBY08): board E10 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Human-in-the-loop online rejection sampling with reward-weighted supervision fine-tunes pi0 stably in 1.5h, beating RL baselines.
- ✅ **VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning** — Lu, Guo, Zhang et al. — [arXiv:2505.18719](https://arxiv.org/abs/2505.18719) — *Talk Mon 9:29, Room 409/410 · **Poster #4707** (session M1LAY07): board E8 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Online RL framework for autoregressive VLAs with VLM process reward model and scaling tricks, improving OpenVLA on LIBERO and real robots.
- ✅ **Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation** — Tang, Jing, Pan et al. — [arXiv:2505.12744](https://arxiv.org/abs/2505.12744) — *Talk Wed 9:41, Room 403/404 · **Poster #1208** (session W1LAY06): board B11 (West face), Exhibit Hall B, Wed 11:00–12:30*  
  LMM directly predicts gripper goals via reasoning; SFT on reasoning data then RL in simulation improves OOD generalization and sim-to-real.
- ✅ **TD-GRPC: Temporal Difference Learning with Group Relative Policy Constraint for Humanoid Locomotion** — Nguyen, Le, Nguyen et al. — [arXiv:2505.13549](https://arxiv.org/abs/2505.13549) — *Talk Mon 9:18, Room 315/316 · **Poster #4123** (session M1LAX03): board B12 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Combines TD-MPC with GRPO-style group-relative ranking and latent trust-region policy constraints for stable humanoid RL.

### Robot Learning & Imitation
- **What Can Robot Foundation Models Learn from Limited Human Demonstrations? Fine-Tuning with Human Task Videos** — Hagane, Goto, Ohama — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22What%20Can%20Robot%20Foundation%20Models%20Learn%20from%20Limited%20Human%20Demonstrations%3F%20Fine-Tuning%20with%20Human%20Task%20Videos%22) — *Talk Mon 9:33, Room 329 · **Poster #421** (session M1FA09): board D8 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Systematic study mixing human task videos into VLA post-training; helps semantic grounding of novel objects but not contact-rich verb-specific motions.
- ✅ **CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations** — Liang, Czempin, Hong et al. — [arXiv:2505.04999](https://arxiv.org/abs/2505.04999) — *Talk Mon 14:36, Room 302/303 · **Poster #3904** (session M2LAX01): board F10 (South face), Exhibit Hall B, Mon 16:30–18:00*  
  Continuous latent action model learned from action-free demos, grounded with a jointly trained decoder on play data, approaching BC with true labels.
- ✅ **Learning from the Best: Smoothness-Driven Metrics for Data Quality in Imitation Learning** — Kulkarni, Dhar, Cui — [arXiv:2604.23000](https://arxiv.org/abs/2604.23000) — *Talk Mon 15:25, Room 302/303 · **Poster #4319** (session M2LBX01): board F13 (East face), Exhibit Hall B, Mon 16:30–18:00*  
  Smoothness metrics score demonstrations, reducing conditional action variance; filtering gives +16% BC success with one-sixth of the data and helps retrieval/reweighting.
- ✅ **Geometric Entropy: When Trajectory Diversity Helps and Hurts in Imitation Learning** — Luo, Liu, Zhou et al. — [arXiv:2606.20871](https://arxiv.org/abs/2606.20871) — *Talk Mon 15:11, Room 302/303 · **Poster #4639** (session M2LAY01): board G11 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Geometric entropy metric reveals an inverted-U between demonstration trajectory diversity and imitation success, shifting with data scale and pretrained VLAs.
- **TRACE: Trajectory Action Chunking Enhancement for Robust Imitation Learning in Robotic Manipulation** — Hu, Luo, Liu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22TRACE%3A%20Trajectory%20Action%20Chunking%20Enhancement%20for%20Robust%20Imitation%20Learning%20in%20Robotic%20Manipulation%22) — *Talk Tue 14:48, Room 317/318 · **Poster #455** (session T2LAX04): board E9 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Finds ACT degrades with larger biased datasets due to spurious correlations; debiasing encoder via action masking/noise fixes it.
- **Improving Policy Learning from Delayed Interventions** — Zhu, Dudek, Meger — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Improving%20Policy%20Learning%20from%20Delayed%20Interventions%22) — *Talk Mon 15:16, Room 329 · **Poster #3237** (session M2FB09): board A9 (South face), Exhibit Hall B, Mon 16:30–18:00*  
  Shows delayed human interventions hurt interactive imitation and learns an intervention-based cost function accounting for delay to recover performance.
- ✅ **Beyond Implicit Force: Evaluating Explicit Force-Torque Proxies in Action Chunking with Transformers** — Wong, Liu, Dayoub — [arXiv:2607.14578](https://arxiv.org/abs/2607.14578) — *Talk Mon 9:12, Room 411/412 · **Poster #3659** (session M1LAX08): board G12 (West face), Exhibit Hall B, Mon 11:00–12:30*  
  Shows ACT's force-awareness comes from leader-follower teleop discrepancy; torque proxies recover contact-aware behavior.

### Diffusion & Flow Policies
- ✅ **VGFM: Expressive Robot Policies Via Dense Value Guidance in Flow Matching** — Koirala, Campbell — [arXiv:2609.14261](https://arxiv.org/abs/2609.14261) — *Talk Tue 14:41, Room 320 · **Poster #3387** (session T2FA02): board G7 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Dense value guidance along flow-matching trajectories enables offline RL policy improvement without BPTT or distillation.
- **Compositional Average Velocity Modeling for Efficient Diffusion Offline Reinforcement Learning** — Nguyen, Syarubany, Kim et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Compositional%20Average%20Velocity%20Modeling%20for%20Efficient%20Diffusion%20Offline%20Reinforcement%20Learning%22) — *Talk Mon 9:23, Room 409/410 · **Poster #4500** (session M1LAY07): board F9 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Single-step flow policy for offline RL via compositional average-velocity modeling with behavior-regularized actor-critic, no BPTT or distillation.

### World Models
- ✅ **RoDyn: Taming Interactive Robot-Dynamic 2.5D World Model for Robotic Manipulation** — Zhang, Wu, Lu et al. — [arXiv:2510.09036](https://arxiv.org/abs/2510.09036) — *Talk Wed 10:04, Room 409/410 · **Poster #1806** (session W1LBX07): board D11 (East face), Exhibit Hall B, Wed 11:00–12:30*  
  2.5D robot-dynamic world model with geometry-aware tokenizer; improves generation fidelity and accelerates MBRL and imitation learning.
- ✅ **Scaling Cross-Embodiment World Models for Dexterous Manipulation** — He, Ai, Mu et al. — [arXiv:2511.01177](https://arxiv.org/abs/2511.01177) — *Talk Wed 9:34, Room 304/305 · **Poster #2451** (session W1LAY02): board B6 (North face), Exhibit Hall B, Wed 11:00–12:30*  
  Particle-based shared action/state representation lets world models scale across human and robot hands for cross-embodiment dexterous control.
- ✅ **Generalizable Robotic Insertion with World Models** — Hansen, Akinola, Guo et al. — [arXiv:2609.28258](https://arxiv.org/abs/2609.28258) — *Talk Mon 9:15, Room 411/412 · **Poster #2858** (session M1LAX08): board G11 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  A single world model trained on up to 90 insertion tasks generalizes zero-shot to unseen parts (56% vs 7% model-free) and scales with objects.
- ✅ **GrndCtrl: Grounding World Models Via Self-Supervised Reward Alignment** — He, Patrikar, Kim et al. — [arXiv:2512.01952](https://arxiv.org/abs/2512.01952) — *Talk Wed 9:46, Room 409/410 · **Poster #4019** (session W1LBX07): board D12 (West face), Exhibit Hall B, Wed 11:00–12:30*  
  RLVR-style self-supervised post-training aligns pretrained video world models with geometric and perceptual verifiable rewards.
- ✅ **ImagiNav: Scalable Embodied Navigation Via Generative Visual Prediction and Inverse Dynamics** — Chen, Cai, Wang et al. — [arXiv:2603.13833](https://arxiv.org/abs/2603.13833) — *Talk Mon 15:25, Room 401/402 · **Poster #4043** (session M2LBX05): board E12 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Navigation via language-conditioned egocentric video generation plus an inverse dynamics model, enabling zero-shot transfer from in-the-wild videos without robot demos.
- ✅ **DAWN: Noise-Robust Quadruped Parkour Via Depth-Denoising World Models** (Award Candidate) — Choi, Kim, Kim et al. — [arXiv:2609.29092](https://arxiv.org/abs/2609.29092) — *Talk Mon 15:16, Room 406 · **Poster #4477** (session M2AB01): board B6 (South face), Exhibit Hall B, Mon 16:30–18:00*  
  World model trained with noisy-to-clean depth reconstruction and contrastive alignment enables filter-free quadruped parkour.

### Generalization, Sim-to-Real & Evaluation
- ✅ **Spotlighting Task-Relevant Features: Object-Centric Representations for Better Generalization in Robotic Manipulation** — Chapin, Machado, Dellandrea et al. — [arXiv:2601.21416](https://arxiv.org/abs/2601.21416) — *Talk Wed 9:11, Room 329 · **Poster #341** (session W1FA09): board A3 (East face), Exhibit Hall B, Wed 11:00–12:30*  
  Large diagnostic study: slot-based object-centric representations greatly improve policy robustness to visual shifts but merge slots under clutter.
- ✅ **LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models** — Hou, Zhao — [arXiv:2603.00592](https://arxiv.org/abs/2603.00592) — *Talk Mon 10:12, Room 401/402 · **Poster #522** (session M1LBY05): board D4 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Shows SOTA VLAs like pi0.5 largely ignore language; builds a semantic-perturbation benchmark and tests augmentation to close the gap.
- ✅ **How VLAs (Really) Work in Open-World Environments** — Rasouli, Wu, Li et al. — [arXiv:2604.21192](https://arxiv.org/abs/2604.21192) — *Talk Tue 15:19, Room 302/303 · **Poster #909** (session T2LBX01): board G12 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Analyzes SOTA VLAs on BEHAVIOR-1K, showing success-rate metrics hide safety violations and inconsistency; proposes safety-aware evaluation protocols.
- ✅ **On the Generalization Capabilities, Design Choices and Limitations of Keypoint Imitation Learning** — Lips, Moletta, Welle et al. — [arXiv:2605.26649](https://arxiv.org/abs/2605.26649) — *Talk Mon 14:30, Room 302/303 · **Poster #2479** (session M2LAX01): board F10 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Systematic real-world study (2000+ rollouts) of keypoint imitation design choices, generalization, and limits versus other representations.
- ✅ **BIFROST: Bridging Invariant Feature Representation for Observation-Space Sim2Real Transfer** — Deng, Hanna — [arXiv:2607.01410](https://arxiv.org/abs/2607.01410) — *Talk Mon 15:49, Room 334 · **Poster #2764** (session M2FB11): board G9 (East face), Exhibit Hall B, Mon 16:30–18:00*  
  Cross-domain bisimulation history encoder on paired sim/real data enables zero-shot sim2real under both visual and dynamics gaps.
- ✅ **The Moving Eye: Enhancing VLA Spatial Generalization Via Hybrid Dynamic Data Collection** — Tang, Zhu, Xie et al. — [arXiv:2607.02322](https://arxiv.org/abs/2607.02322) — *Talk Mon 9:35, Room 411/412 · **Poster #4054** (session M1LAY08): board F11 (West face), Exhibit Hall B, Mon 11:00–12:30*  
  Shows VLAs learn spatial shortcuts from fixed camera poses; dynamic moving-camera data collection improves spatial generalization.
- ✅ **Abstract Sim2Real through Approximate Information States** — Deng, Li, Hanna — [arXiv:2604.15289](https://arxiv.org/abs/2604.15289) · [RA-L](https://doi.org/10.1109/lra.2026.3688068) — *Talk Tue 14:45, Room 409/410 · **Poster #4942** (session T2LAX07): board E7 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  Formalizes sim2real with abstract simulators via state abstraction; shows history-conditioned grounding with real data enables transfer.
- ✅ **REALM: A Real-To-Sim Validated Benchmark for Generalization in Robotic Manipulation** — Sedlacek, Yefanov, Ponimatkin et al. — [arXiv:2512.19562](https://arxiv.org/abs/2512.19562) · [RA-L](https://doi.org/10.1109/lra.2026.3692088) — *Talk Wed 15:04, Room 317/318 · **Poster #4998** (session W2LAY04): board C12 (East face), Exhibit Hall B, Wed 16:30–18:00*  
  Real-to-sim validated benchmark with 15 perturbation factors showing pi0, pi0-FAST, GR00T generalization gaps with sim-real correlation.

### Reward Learning & Foundation-Model Rewards
- ✅ **Generalizable Dense Reward for Long-Horizon Robotic Tasks** — Yong, Sheng, Qi et al. — [arXiv:2604.00055](https://arxiv.org/abs/2604.00055) — *Talk Mon 9:22, Room 320 · **Poster #3735** (session M1FA02): board F2 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Dense rewards for RL fine-tuning foundation policies: VLM progress for value init plus self-certainty intrinsic reward, beating RL fine-tuning baselines.
- **IRL4IDM: Inverse Reinforcement Learning for Inverse Dynamics Model in Video Generative Robot Planner** — Huang, Zhu, Ren et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22IRL4IDM%3A%20Inverse%20Reinforcement%20Learning%20for%20Inverse%20Dynamics%20Model%20in%20Video%20Generative%20Robot%20Planner%22) — *Talk Tue 15:31, Room 409/410 · **Poster #4073** (session T2LBX07): board E6 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  Learns inverse dynamics for video-generative planners via IRL: dense rewards from video temporal alignment and online goal-conditioned RL, no action labels.

## P1 — Should Read

### VLA Models
- ✅ **ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning** — Yang, Shi, Mingyuan et al. — [arXiv:2607.12992](https://arxiv.org/abs/2607.12992) — *Talk Mon 9:46, Room 411/412 · **Poster #271** (session M1LBX08): board G10 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Seam-aware training for chunked VLA policies with continuity losses, history corruption, and AWAC fine-tuning to reduce chunk-boundary jitter.
- ✅ **FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models** — Lin, Duan, Fang et al. — [arXiv:2510.01642](https://arxiv.org/abs/2510.01642) — *Talk Mon 15:45, Room 411/412 · **Poster #725** (session M2LBY08): board B10 (South face), Exhibit Hall B, Mon 16:30–18:00*  
  Automatically generates failure cases with executable recovery actions; a VLM trained on them improves pi0-FAST/OpenVLA recovery by up to 22.6%.
- ✅ **GeoVLA: Empowering 3D Representations in Vision-Language-Action Models** (Award Candidate) — Sun, Xie, Liu et al. — [arXiv:2508.09071](https://arxiv.org/abs/2508.09071) — *Talk Tue 9:33, Room 406 · **Poster #888** (session T1AA01): board B7 (East face), Exhibit Hall B, Tue 11:00–12:30*  
  Adds point-cloud embeddings and a 3D-aware action expert to a VLA, improving LIBERO/ManiSkill2 and viewpoint robustness.
- ✅ **LLaDA-VLA: Vision Language Diffusion Action Models** — Wen, Li, Gu et al. — [arXiv:2509.06932](https://arxiv.org/abs/2509.06932) — *Talk Wed 14:31, Room 411/412 · **Poster #1635** (session W2LAX08): board E9 (South face), Exhibit Hall B, Wed 16:30–18:00*  
  Builds a VLA on masked-diffusion VLMs with action-token classification and hierarchical decoding, outperforming autoregressive VLAs.
- ✅ **HapticVLA: Contact-Rich Manipulation Via Vision-Language-Action Model without Inference-Time Tactile Sensing** — Gubernatorov, Sannikov, Mikhalchuk et al. — [arXiv:2603.15257](https://arxiv.org/abs/2603.15257) — *Talk Mon 15:42, Room 302/303 · **Poster #3082** (session M2LBY01): board G12 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Safety-aware reward-weighted flow matching with tactile rewards, then distills a tactile token so VLA needs no tactile sensors at inference.
- **BeliefVLA: Implicit Belief-State Driven Vision-Language-Action Models for Robust Manipulation** — Liu, Huang, Han et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22BeliefVLA%3A%20Implicit%20Belief-State%20Driven%20Vision-Language-Action%20Models%20for%20Robust%20Manipulation%22) — *Talk Wed 15:01, Room 411/412 · **Poster #3638** (session W2LAY08): board D9 (North face), Exhibit Hall B, Wed 16:30–18:00*  
  Injects V-JEPA2 implicit belief states from history into VLAs to add dynamics-aware priors and temporal memory.

### RL Fine-Tuning & Policy Optimization
- ✅ **Keyframe-Guided Structured Rewards for Reinforcement Learning in Long-Horizon Laboratory Robotics** — Qiu, Sun, Ye et al. — [arXiv:2603.00719](https://arxiv.org/abs/2603.00719) — *Talk Tue 15:19, Room 409/410 · **Poster #892** (session T2LBX07): board D7 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  Keyframe-derived progress rewards drive human-in-the-loop RL fine-tuning of a VLA, reaching 82% on lab tasks vs 42-47% for HG-DAgger/ConRFT.
- ✅ **Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning** — Zhang, Weng, Liu et al. — [arXiv:2607.13818](https://arxiv.org/abs/2607.13818) — *Talk Mon 9:06, Room 411/412 · **Poster #1756** (session M1LAX08): board G11 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Agentic RL policy monitors VLA execution quality metrics and selects recovery modes, boosting LIBERO robustness under disturbances.
- ✅ **DexHiL: A Human-In-The-Loop Framework for Post-Training VLA Models in Dexterous Manipulation** — Han, Chen, Xu et al. — [arXiv:2603.09121](https://arxiv.org/abs/2603.09121) — *Talk Tue 15:49, Room 328 · **Poster #1815** (session T2FB08): board C8 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  Human-in-the-loop arm-hand intervention framework with intervention-aware sampling for post-training dexterous VLAs, +25% over offline.
- ✅ **HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual Reinforcement Learning** — Liu, Mai, He et al. — [arXiv:2606.22860](https://arxiv.org/abs/2606.22860) — *Talk Tue 15:34, Room 409/410 · **Poster #2243** (session T2LBX07): board F5 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Model-agnostic residual RL adapter on VLA actions with human-in-the-loop guidance reaches >95% real success in 1.5 hours.
- ✅ **NavCMPO: Critic-Guided MeanFlow Policy Optimization for Adaptive Navigation** — An, Wu, Liu et al. — [arXiv:2607.14643](https://arxiv.org/abs/2607.14643) — *Talk Wed 9:03, Room 409/410 · **Poster #2490** (session W1LAX07): board C13 (South face), Exhibit Hall B, Wed 11:00–12:30*  
  Few-step MeanFlow navigation policy with critic-gradient trajectory refinement, then PPO fine-tuning with BC regularization for embodiment adaptation.
- **HIL-ResRFT: Improving Imitation Learning Policy with Human in the Loop Residual RL for Robot Manipulation** — Li, Zhou, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22HIL-ResRFT%3A%20Improving%20Imitation%20Learning%20Policy%20with%20Human%20in%20the%20Loop%20Residual%20RL%20for%20Robot%20Manipulation%22) — *Talk Tue 9:03, Room 302/303 · **Poster #4035** (session T1LAX01): board B5 (East face), Exhibit Hall B, Tue 11:00–12:30*  
  Human-in-the-loop residual RL fine-tunes imitation policies in the real world for robustness and sample efficiency.

### Robot Learning & Imitation
- **Mitigating Perceptual Interference in Multi-Stage Visuomotor Imitation with Stage-Conditioned Spatial Gating** — Zhang, Wang, Chen — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Mitigating%20Perceptual%20Interference%20in%20Multi-Stage%20Visuomotor%20Imitation%20with%20Stage-Conditioned%20Spatial%20Gating%22) — *Talk Tue 10:19, Room 329 · **Poster #2113** (session T1FB09): board A1 (East face), Exhibit Hall B, Tue 11:00–12:30*  
  Identifies perceptual interference in multi-stage imitation and adds stage-conditioned spatial gating in the encoder, improving late-stage success.
- ✅ **RoboSSM: Scalable In-Context Imitation Learning Via State-Space Models** — Yoo, Hu, Zhu et al. — [arXiv:2509.19658](https://arxiv.org/abs/2509.19658) — *Talk Mon 9:35, Room 409/410 · **Poster #3133** (session M1LAY07): board E8 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Replaces Transformers with state-space models for in-context imitation learning, extrapolating to longer demonstration prompts.
- ✅ **SAIL: Test-Time Scaling for In-Context Imitation Learning with VLM** — Sato, Iwasawa, Tang et al. — [arXiv:2603.08269](https://arxiv.org/abs/2603.08269) — *Talk Tue 15:49, Room 320 · **Poster #3510** (session T2FB02): board G6 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  Test-time scaling for in-context imitation via MCTS over trajectory refinements scored by a VLM.
- ✅ **Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition** — Quartey, Castro, Rosen et al. — [arXiv:2605.20648](https://arxiv.org/abs/2605.20648) — *Talk Mon 9:58, Room 411/412 · **Poster #3773** (session M1LBX08): board E11 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Policies jointly generating actions and predicate beliefs improve both and enable zero-shot skill composition via symbolic planning.
- **Augmenting Robot Imitation Learning with Human Videos without Additional Robot Data** — Zhang, Xiong, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Augmenting%20Robot%20Imitation%20Learning%20with%20Human%20Videos%20without%20Additional%20Robot%20Data%22) — *Talk Mon 10:19, Room 329 · **Poster #4124** (session M1FB09): board D7 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Augments fixed-budget robot imitation with unpaired human videos via hand-keypoint to joint-space mapping and cross-domain feature alignment.
- ✅ **Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning** — Krohn, Prasad, Tiboni et al. — [arXiv:2511.14427](https://arxiv.org/abs/2511.14427) · [RA-L](https://doi.org/10.1109/lra.2026.3681156) — *Talk Mon 9:57, Room 320 · **Poster #4929** (session M1FB02): board G2 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Masked multisensory autoencoder pretraining gives robust representations for contact-rich RL.

### Diffusion & Flow Policies
- ✅ **Amortizing Trajectory Diffusion with Keyed Drift Fields** — Puthumanaillam, Ornik — [arXiv:2603.14056](https://arxiv.org/abs/2603.14056) — *Talk Mon 9:00, Room 321 · **Poster #3205** (session M1FA03): board A10 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  One-step trajectory generator trained with a drift-field objective in a condition-aware key space, matching diffusion planning at far lower latency.
- ✅ **SUREFlow: State-Space Uncertainty-Aware REsidual Flow Matching for Robust Robot Manipulation** — Islam, Peddapalli, Lee et al. — [arXiv:2607.10504](https://arxiv.org/abs/2607.10504) — *Talk Wed 14:52, Room 328 · **Poster #726** (session W2FA08): board E8 (South face), Exhibit Hall B, Wed 16:30–18:00*  
  Residual flow-matching policy that predicts input-dependent uncertainty to selectively refine unreliable action dimensions, strong LIBERO results at 179M params.
- ✅ **HiFlow: Tokenization-Free Scale-Wise Autoregressive Policy Learning Via Flow Matching** — Yashima, Seno, Kurita et al. — [arXiv:2603.27281](https://arxiv.org/abs/2603.27281) — *Talk Mon 9:32, Room 411/412 · **Poster #1332** (session M1LAY08): board F11 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Tokenization-free coarse-to-fine autoregressive flow policy using temporally pooled multi-scale action targets, beating diffusion and tokenized AR policies.
- **SANE: Smooth Action Via Noise Envelope Conditioned on Gaussian Process for Denoising Policies** — Lv, Tan, Zhao et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22SANE%3A%20Smooth%20Action%20Via%20Noise%20Envelope%20Conditioned%20on%20Gaussian%20Process%20for%20Denoising%20Policies%22) — *Talk Wed 15:25, Room 411/412 · **Poster #1372** (session W2LBX08): board C9 (East face), Exhibit Hall B, Wed 16:30–18:00*  
  Replaces i.i.d. flow-matching noise with a Gaussian-process noise prior conditioned on executed actions for smooth inter-chunk VLA actions.
- ✅ **FLUX: Accelerating Cross-Embodiment Generative Navigation Policies Via Rectified Flow and Static-To-Dynamic Learning** — Gong, Zhong, DING et al. — [arXiv:2603.12806](https://arxiv.org/abs/2603.12806) — *Talk Wed 9:49, Room 409/410 · **Poster #2103** (session W1LBX07): board D12 (North face), Exhibit Hall B, Wed 11:00–12:30*  
  Rectified-flow navigation policy pretrained then RL-refined in crowds; stochastic RL-learned recovery aids generalization, zero-shot across wheeled, quadruped, humanoid.
- ✅ **D3P: Dynamic Denoising Diffusion Policy Via Reinforcement Learning** — Yu, Gao, Wu et al. — [arXiv:2508.06804](https://arxiv.org/abs/2508.06804) — *Talk Tue 15:27, Room 320 · **Poster #2131** (session T2FB02): board G6 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  RL-trained adapter allocates denoising steps per action at test time, exploiting that only some actions are critical.
- ✅ **RoamFlow: Reinforcement-Aligned One-Step Action MeanFlow Policy for Image-Goal Navigation** — ZHANG, Chen, Gao et al. — [arXiv:2606.29934](https://arxiv.org/abs/2606.29934) — *Talk Wed 9:00, Room 409/410 · **Poster #2356** (session W1LAX07): board C13 (East face), Exhibit Hall B, Wed 11:00–12:30*  
  One-step MeanFlow navigation policy initialized by imitation then refined with RL for image-goal navigation.
- ✅ **From Flow to One Step: Real-Time Multi-Modal Trajectory Policies Via Implicit Maximum Likelihood Estimation-Based Distribution Distillation** — Dong, Zhang, Zhang et al. — [arXiv:2603.09415](https://arxiv.org/abs/2603.09415) — *Talk Wed 15:28, Room 411/412 · **Poster #2814** (session W2LBX08): board C9 (South face), Exhibit Hall B, Wed 16:30–18:00*  
  Distills flow-matching policy into a one-step student via IMLE with Chamfer set loss, preserving multimodality at 125 Hz.
- **Learning Consistent Manipulation Via Joint Action–Object Diffusion** — Yu, Fan — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Learning%20Consistent%20Manipulation%20Via%20Joint%20Action%E2%80%93Object%20Diffusion%22) — *Talk Tue 9:58, Room 302/303 · **Poster #2981** (session T1LBX01): board A4 (West face), Exhibit Hall B, Tue 11:00–12:30*  
  Diffusion policy jointly denoising robot and object trajectories enforces action-outcome consistency and improves manipulation robustness.
- ✅ **Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration** — Miller, Jagersand — [arXiv:2607.29482](https://arxiv.org/abs/2607.29482) — *Talk Mon 15:31, Room 302/303 · **Poster #3033** (session M2LBX01): board F13 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Stochastic-interpolant policy initialized from recent action history instead of Gaussian noise, cutting transport cost ~10x and latency while matching success.
- ✅ **CoRDE: Concept-Prior Routed Diffusion Experts for Structural Generalization in Robot Manipulation** — Huang, Zhao, Zhou et al. — [arXiv:2606.21935](https://arxiv.org/abs/2606.21935) — *Talk Mon 14:33, Room 302/303 · **Poster #3932** (session M2LAX01): board F10 (North face), Exhibit Hall B, Mon 16:30–18:00*  
  Concept-prior routed LoRA diffusion experts avoid MoE routing collapse and gradient conflict in multi-task long-horizon imitation.
- ✅ **Trajectory-Consistent Flow Matching for Robust Visuomotor Policy Learning** — Ahmed, Nag, Akash et al. — [arXiv:2605.08511](https://arxiv.org/abs/2605.08511) — *Talk Mon 9:38, Room 409/410 · **Poster #3959** (session M1LAY07): board F8 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Closes flow-matching train-inference gap with multi-step trajectory consistency training, velocity smoothness regularization and RK4 integration.
- ✅ **3D FlowMatch Actor: Unified 3D Policy for Single and Dual-Arm Manipulation** — Gkanatsios, Xu, Bronars et al. — [arXiv:2508.11002](https://arxiv.org/abs/2508.11002) — *Talk Tue 10:15, Room 302/303 · **Poster #4030** (session T1LBY01): board B3 (South face), Exhibit Hall B, Tue 11:00–12:30*  
  3D flow-matching policy with relative attention achieving 30x faster training/inference and large SOTA gains on bimanual PerAct2 and RLBench.

### World Models
- ✅ **DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation Via World Model** — Yin, Lai — [arXiv:2608.22278](https://arxiv.org/abs/2608.22278) — *Talk Mon 15:08, Room 401/402 · **Poster #279** (session M2LAY05): board D13 (East face), Exhibit Hall B, Mon 16:30–18:00*  
  Uses an RSSM world model as predictive representation and multi-step supervision to distill privileged teachers into visual humanoid loco-manipulation policies.
- **DreamManip: Holistic Visual Planning for Long-Horizon Robotic Manipulation via Video Foundation Models** — Xiong, Zhong, Fan et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22DreamManip%3A%20Holistic%20Visual%20Planning%20for%20Long-Horizon%20Robotic%20Manipulation%20via%20Video%20Foundation%20Models%22) — *Talk Wed 15:45, Room 411/412 · **Poster #1216** (session W2LBY08): board C10 (West face), Exhibit Hall B, Wed 16:30–18:00*  
  Video foundation model imagines the full visual path from current to goal image, then executes it for long-horizon manipulation.
- ✅ **RoboDream: Compositional World Models for Scalable Robot Data Synthesis** — Ye, Xue, Van Hoorick et al. — [arXiv:2606.02577](https://arxiv.org/abs/2606.02577) — *Talk Wed 15:51, Room 411/412 · **Poster #3902** (session W2LBY08): board C11 (East face), Exhibit Hall B, Wed 16:30–18:00*  
  Embodiment-centric video world model synthesizes demonstrations in new scenes/objects/viewpoints, improving downstream policies with less real data.
- ✅ **Cosmos-Surg-dVRK: World Foundation Model-Based Automated Online Evaluation of Surgical Robot Policy Learning** — Zbinden, Nelson, Chen et al. — [arXiv:2510.16240](https://arxiv.org/abs/2510.16240) · [RA-L](https://doi.org/10.1109/lra.2026.3675962) — *Talk Tue 9:07, Room 409/410 · **Poster #4744** (session T1LAX07): board E7 (South face), Exhibit Hall B, Tue 11:00–12:30*  
  Fine-tunes the Cosmos world foundation model to evaluate surgical robot policies online in place of physical rollouts.

### Generalization, Sim-to-Real & Evaluation
- ✅ **AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models** — Heo, Woo, Kim et al. — [arXiv:2603.05868](https://arxiv.org/abs/2603.05868) — *Talk Mon 10:09, Room 401/402 · **Poster #359** (session M1LBY05): board D4 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Zero-shot viewpoint robustness for VLAs by re-rendering test camera views to the training viewpoint with feed-forward novel view synthesis.
- ✅ **Attentive Feature Aggregation Or: How Policies Learn to Stop Worrying about Robustness and Attend to Task-Relevant Visual Cues** — Tsagkas, Sochopoulos, Danier et al. — [arXiv:2511.10762](https://arxiv.org/abs/2511.10762) — *Talk Mon 9:23, Room 411/412 · **Poster #462** (session M1LAY08): board F11 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Trainable attentive pooling over pretrained visual features makes visuomotor policies robust to distractors without augmentation or PVR fine-tuning.
- ✅ **DADiff: Diffusion-Driven Cross-Domain Policy Adaptation for Reinforcement Learning** — Chen, Satheesh, Da et al. — [arXiv:2607.16090](https://arxiv.org/abs/2607.16090) — *Talk Mon 9:11, Room 320 · **Poster #800** (session M1FA02): board F2 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Uses diffusion generative trajectory discrepancies to estimate dynamics mismatch for reward modification or data selection in cross-domain RL, with bounds.
- ✅ **Red-Teaming Vision-Language-Action Models Via Quality Diversity Prompt Generation for Robust Robot Policies** — Srikanth, Liang, Hsu et al. — [arXiv:2603.12510](https://arxiv.org/abs/2603.12510) — *Talk Tue 9:33, Room 329 · **Poster #1795** (session T1FA09): board A2 (East face), Exhibit Hall B, Tue 11:00–12:30*  
  Quality-diversity red-teaming generates diverse task-relevant instructions that make VLAs fail, then uses them to improve robustness.
- **Scene2Gym: Transforming Real-World Scenes into Interactive Training Environments for Robot Learning** — Ji, Chen, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Scene2Gym%3A%20Transforming%20Real-World%20Scenes%20into%20Interactive%20Training%20Environments%20for%20Robot%20Learning%22) — *Talk Wed 9:06, Room 411/412 · **Poster #2019** (session W1LAX08): board E7 (South face), Exhibit Hall B, Wed 11:00–12:30*  
  Builds metric-aligned 3DGS digital twins from real scenes; IL and visual RL policies trained there transfer zero-shot to real.
- **SRDR: Smoothed Return-Guided Domain Randomization for Stable and Robust Policy Learning** — Wang, Ghoreishi — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22SRDR%3A%20Smoothed%20Return-Guided%20Domain%20Randomization%20for%20Stable%20and%20Robust%20Policy%20Learning%22) — *Talk Mon 14:42, Room 411/412 · **Poster #3353** (session M2LAX08): board A13 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Variance-reduced smoothed-return adaptive domain randomization for stable, robust policy learning across seeds.
- ✅ **ZeroBot: Learning from Scratch in Minutes with Generative Real2Sim** — Kapelyukh, Zhang, James et al. — [arXiv:2609.34010](https://arxiv.org/abs/2609.34010) · [RA-L](https://doi.org/10.1109/lra.2026.3662595) — *Talk Tue 10:24, Room 401/402 · **Poster #5050** (session T1LBY05): board C3 (North face), Exhibit Hall B, Tue 11:00–12:30*  
  Image-to-3D generative real2sim plus contact-sampling action space enables learning real manipulation from scratch via parallel RL in minutes.

### Reward Learning & Foundation-Model Rewards
- **Warming-Up and Shaping Robot Policy Learning with Large Language Model-Based Critic Feedback** — Hayamizu, Ai, hou et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Warming-Up%20and%20Shaping%20Robot%20Policy%20Learning%20with%20Large%20Language%20Model-Based%20Critic%20Feedback%22) — *Talk Tue 15:16, Room 409/410 · **Poster #1076** (session T2LBX07): board D7 (South face), Exhibit Hall B, Tue 16:30–18:00*  
  Uses LLMs as external critics to initialize and shape value estimates, improving RL sample efficiency over LLM-generated policies.
- **SPEAR: Selective Preference Elicitation with Adaptive Rewards for VLM-Guided Robotic Manipulation** — Wu, Liu, Zhou et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22SPEAR%3A%20Selective%20Preference%20Elicitation%20with%20Adaptive%20Rewards%20for%20VLM-Guided%20Robotic%20Manipulation%22) — *Talk Mon 10:12, Room 411/412 · **Poster #1793** (session M1LBY08): board F10 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Makes VLMs efficient preference oracles via distinguishability-aware query selection and adaptive reward transition, learning real Mobile ALOHA tasks online.
- ✅ **Learning Acrobatic Flight from Preferences** — Merk, Geles, Xing et al. — [arXiv:2508.18817](https://arxiv.org/abs/2508.18817) — *Talk Tue 14:34, Room 315/316 · **Poster #2034** (session T2LAX03): board B5 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  Distributional reward ensembles model per-step uncertainty in preference-based RL, reaching 88% of shaped-reward performance for real acrobatic flight.
- ✅ **Stage-Transition Dense Reward Modeling for Reinforcement Learning** — yang, Chen, Wang et al. — [arXiv:2606.31377](https://arxiv.org/abs/2606.31377) — *Talk Tue 15:28, Room 409/410 · **Poster #2333** (session T2LBX07): board E6 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Converts expert videos into stage-transition and within-stage progress dense rewards with OOD detection to train RL from scratch.
- ✅ **Training Fast Robot Policies with Slow Foundation Models** — Singh, Bhattacharyya, Namboodiri et al. — [arXiv:2406.05881](https://arxiv.org/abs/2406.05881) — *Talk Tue 15:38, Room 320 · **Poster #4072** (session T2FB02): board G6 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  LLM-written reward code refined by VLM diagnosis of failed rollouts trains fast policies without foundation models at deployment.
- ✅ **PrefMoE: Robust Preference Modeling with Mixture-Of-Experts Reward Learning** — Yuan, Wang, Zhao et al. — [arXiv:2605.00384](https://arxiv.org/abs/2605.00384) — *Talk Tue 10:18, Room 403/404 · **Poster #4214** (session T1LBY06): board A10 (West face), Exhibit Hall B, Tue 11:00–12:30*  
  Mixture-of-experts reward model with soft trajectory routing handles heterogeneous, conflicting preference labels for more robust preference-based RL.
- ✅ **MotionVL: Vision-Language Supervision for Reinforcement Learning of Humanoid Motion** — Luo, Wu, Xiong et al. — [RA-L](https://doi.org/10.1109/lra.2026.3669793) — *Talk Mon 15:05, Room 401/402 · **Poster #4738** (session M2LAY05): board E11 (East face), Exhibit Hall B, Mon 16:30–18:00*  
  VLM describes humanoid behavior and LLM generates/refines rewards in closed loop for humanoid motion RL.

### RL Algorithms & Exploration
- ✅ **FastDSAC: Enhancing Policy Plasticity Via Constrained Exploration for Scalable Humanoid Locomotion** — Lu, Dun, zhou et al. — [arXiv:2606.31691](https://arxiv.org/abs/2606.31691) — *Talk Mon 9:03, Room 315/316 · **Poster #2192** (session M1LAX03): board B11 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Truncated-Gaussian distributional SAC for massively parallel high-UTD training preserves plasticity and speeds humanoid locomotion learning.
- ✅ **CDE: Concept-Driven Exploration for Reinforcement Learning** — Mao, Liu, Zabounidis et al. — [arXiv:2510.08851](https://arxiv.org/abs/2510.08851) — *Talk Tue 15:25, Room 409/410 · **Poster #3322** (session T2LBX07): board E6 (South face), Exhibit Hall B, Tue 16:30–18:00*  
  VLM-generated object concepts reconstructed via an auxiliary loss provide intrinsic rewards for targeted exploration in visual RL.
- ✅ **Where-To-Learn: Analytical Policy Gradient Directed Exploration for On-Policy Robotic Reinforcement Learning** — Chang, Yao, Liu et al. — [arXiv:2603.27317](https://arxiv.org/abs/2603.27317) — *Talk Mon 15:38, Room 320 · **Poster #4952** (session M2FB02): board D4 (North face), Exhibit Hall B, Mon 16:30–18:00*  
  Directed exploration for PPO using analytical policy gradients from differentiable dynamics to steer toward high-reward regions.
- ✅ **LEACL: LLM-Enhanced Automatic Curriculum Learning for Reinforcement Learning in Long-Horizon Manipulation Tasks** — Heravi, Ouyang, Xu et al. — [arXiv:2607.23515](https://arxiv.org/abs/2607.23515) — *Talk Tue 15:22, Room 409/410 · **Poster #1508** (session T2LBX07): board E6 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  Uses LLMs to automatically construct curricula for RL on long-horizon sparse-reward manipulation.
- ✅ **E2HiL: Entropy-Guided Sample Selection for Efficient Real-World Human-In-The-Loop Reinforcement Learning** — Deng, Lin, XUE et al. — [arXiv:2601.19969](https://arxiv.org/abs/2601.19969) — *Talk Mon 10:27, Room 411/412 · **Poster #5002** (session M1LBY08): board E10 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Selects human-in-the-loop RL samples via influence on policy entropy, improving success 25% with fewer interventions on real robots.

### Continual Learning, Adaptation & Cross-Embodiment
- **Collective Learning for Unified Robotic Manipulation Via Predictive Dynamics Adaption** — Liu, Meng, Bing et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Collective%20Learning%20for%20Unified%20Robotic%20Manipulation%20Via%20Predictive%20Dynamics%20Adaption%22) — *Talk Mon 14:42, Room 302/303 · **Poster #839** (session M2LAX01): board F11 (East face), Exhibit Hall B, Mon 16:30–18:00*  
  Separates shared task intent from embodiment dynamics via a predictive-dynamics latent regularizer, avoiding collapse when pooling multi-robot data.
- ✅ **RoboHarness: A Memory-Augmented Policy Harness for Vision-Language-Action Model Robustness Via In-Context Adaptation** — Li, Li, Zhou et al. — [arXiv:2603.24060](https://arxiv.org/abs/2603.24060) — *Talk Wed 9:32, Room 403/404 · **Poster #1273** (session W1LAY06): board A12 (South face), Exhibit Hall B, Wed 11:00–12:30*  
  Memory-augmented harness with retrieval and failure attribution lets frozen VLAs adapt in context to OOD perturbations.
- ✅ **LIDEA: Human-To-Robot Imitation Learning Via Implicit Feature Distillation and Explicit Geometry Alignment** — Xu, Lin, Zhan et al. — [arXiv:2604.10677](https://arxiv.org/abs/2604.10677) — *Talk Tue 15:08, Room 317/318 · **Poster #2048** (session T2LAY04): board F9 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  Learns policies from human videos via latent distillation aligning human/robot features and embodiment-agnostic 3D geometry; replaces 80% robot demos.
- **Relational Dexterity: Transferring Human Dexterity Via Relational Geometric Prior** — Chen, Zhang, Liu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Relational%20Dexterity%3A%20Transferring%20Human%20Dexterity%20Via%20Relational%20Geometric%20Prior%22) — *Talk Wed 9:58, Room 304/305 · **Poster #2405** (session W1LBX02): board B5 (South face), Exhibit Hall B, Wed 11:00–12:30*  
  Embodiment-invariant relational geometric prior from human hand data guides curriculum RL for dexterous transfer.
- **Intra-Rollout Self-Supervised Adaptation of Inverse Dynamics Models for Sub-Centimeter Precision Placement** — Xu, Sun, Yi et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Intra-Rollout%20Self-Supervised%20Adaptation%20of%20Inverse%20Dynamics%20Models%20for%20Sub-Centimeter%20Precision%20Placement%22) — *Talk Wed 9:13, Room 401/402 · **Poster #2538** (session W1LAX05): board C9 (North face), Exhibit Hall B, Wed 11:00–12:30*  
  Adapts inverse dynamics models online within a rollout using self-generated transitions plus decaying stochastic excitation to reach sub-centimeter placement.
- ✅ **CEI: A Unified Interface for Cross-Embodiment Visuomotor Policy Learning in 3D Space** — Wu, Li, Gong et al. — [arXiv:2601.09163](https://arxiv.org/abs/2601.09163) · [RA-L](https://doi.org/10.1109/lra.2026.3656802) — *Talk Mon 14:48, Room 302/303 · **Poster #4882** (session M2LAX01): board F11 (South face), Exhibit Hall B, Mon 16:30–18:00*  
  Cross-embodiment interface aligning trajectories by functional similarity to transfer demos across arms and end-effectors.

### Humanoid & Legged Locomotion via RL
- ✅ **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation** (Award Candidate) — He, Xu, Li et al. — [arXiv:2603.03279](https://arxiv.org/abs/2603.03279) — *Talk Mon 9:11, Room 406 · **Poster #1024** (session M1AA01): board B7 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Physics-based retargeting, distilled latent skill controller and RL fine-tuning enable goal-conditioned humanoid loco-manipulation from egocentric perception.
- ✅ **DreamControl-V2: Simpler and Scalable Autonomous Humanoid Skills Via Trainable Guided Diffusion Priors** — Harithas, Kwak, Katara et al. — [arXiv:2604.00202](https://arxiv.org/abs/2604.00202) — *Talk Tue 9:46, Room 320 · **Poster #1647** (session T1FB02): board F4 (South face), Exhibit Hall B, Tue 11:00–12:30*  
  Trains guided diffusion motion priors directly in robot space to guide RL for humanoid loco-manipulation skills.
- ✅ **APEX: Action Priors Enable Efficient Exploration for Robust Motion Tracking on Legged Robots** — Sood, Nakhwa, SUN et al. — [arXiv:2511.09091](https://arxiv.org/abs/2511.09091) — *Talk Wed 14:58, Room 304/305 · **Poster #2080** (session W2LAY02): board C6 (West face), Exhibit Hall B, Wed 16:30–18:00*  
  Decaying action priors from demonstrations guide early RL exploration for motion tracking, yielding pure RL policy without reference inputs.
- **Role-Based Reward Decomposition for Legged Locomotion Reinforcement Learning** — Tian, Trahanias — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Role-Based%20Reward%20Decomposition%20for%20Legged%20Locomotion%20Reinforcement%20Learning%22) — *Talk Mon 15:27, Room 320 · **Poster #2535** (session M2FB02): board D4 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Careful study of stance/swing role-based reward decomposition with multi-critic variants; shows when decomposition helps or hurts under objective conflict.
- ✅ **GMT: General Motion Tracking for Humanoid Whole-Body Control** — Chen, Ji, Cheng et al. — [arXiv:2506.14770](https://arxiv.org/abs/2506.14770) — *Talk Tue 10:08, Room 320 · **Poster #2995** (session T1FB02): board F4 (North face), Exhibit Hall B, Tue 11:00–12:30*  
  Single unified humanoid motion-tracking policy using adaptive motion sampling and motion MoE for real-world whole-body control.
- **Learning Humanoid Agile and Contact-Rich Control from Latent Predictive Motion Priors** — He, Dong, Zhang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Learning%20Humanoid%20Agile%20and%20Contact-Rich%20Control%20from%20Latent%20Predictive%20Motion%20Priors%22) — *Talk Mon 9:09, Room 315/316 · **Poster #3810** (session M1LAX03): board B11 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Latent predictive motion prior from task-agnostic human motion replaces heavy reward shaping for agile humanoid RL.

### Data Generation & Collection
- ✅ **CRAFT: Video Diffusion for Bimanual Robot Data Generation** — Chen, Liu, Sukhatme et al. — [arXiv:2604.03552](https://arxiv.org/abs/2604.03552) — *Talk Tue 14:39, Room 317/318 · **Poster #1016** (session T2LAX04): board D9 (South face), Exhibit Hall B, Tue 16:30–18:00*  
  Uses Canny-conditioned video diffusion to turn sim trajectories into diverse photorealistic bimanual demos with action labels.
- ✅ **RADAR: Closed-Loop Robotic Data Generation Via Semantic Planning and Autonomous Causal Environment Reset** — Wang, Zhu, Zhong et al. — [arXiv:2603.11811](https://arxiv.org/abs/2603.11811) — *Talk Tue 9:00, Room 302/303 · **Poster #2233** (session T1LAX01): board B5 (North face), Exhibit Hall B, Tue 11:00–12:30*  
  Fully autonomous closed-loop data engine with VLM task planning and causal environment resets from 2-5 demonstrations.
- ✅ **Scalable Multi-Task Data Generation Via Reinforcement Learning for Language-Conditioned Bimanual Dexterous Manipulation** — Li, Jin, Liu et al. — [arXiv:2606.22471](https://arxiv.org/abs/2606.22471) — *Talk Tue 10:24, Room 302/303 · **Poster #3230** (session T1LBY01): board A3 (East face), Exhibit Hall B, Tue 11:00–12:30*  
  RL-based pipeline with generalizable rewards and domain randomization to synthesize language-conditioned bimanual dexterous datasets.

### Additional
- **CRISP: Context-Robust Skill Inference for Long-Horizon Offline Robotic Control** — Zaidi, Munir, Zaidi — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22CRISP%3A%20Context-Robust%20Skill%20Inference%20for%20Long-Horizon%20Offline%20Robotic%20Control%22) — *Talk Mon 9:26, Room 409/410 · **Poster #1431** (session M1LAY07): board E8 (West face), Exhibit Hall B, Mon 11:00–12:30*  
  Hierarchical offline RL with masked latent skill inference, diffusion skill execution and conservative latent values; robust to missing context.
- ✅ **Learning Hierarchical Skill Policies with Offline Quality-Diversity Reinforcement Learning** — Anakewat, Osa, Harada — [arXiv:2608.19684](https://arxiv.org/abs/2608.19684) — *Talk Mon 10:19, Room 320 · **Poster #2223** (session M1FB02): board G2 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Advantage-weighted quality-diversity offline skill extraction plus dataset reuse for offline-to-online hierarchical RL in sparse-reward tasks.
- ✅ **VLASH: Real-Time VLAs Via Future-State-Aware Asynchronous Inference** — Tang, Sun, Zhao et al. — [arXiv:2512.01031](https://arxiv.org/abs/2512.01031) — *Talk Wed 9:26, Room 403/404 · **Poster #1611** (session W1LAY06): board A12 (North face), Exhibit Hall B, Wed 11:00–12:30*  
  Future-state-aware asynchronous inference lets VLAs act continuously without stalls, improving reaction latency on dynamic tasks.
- ✅ **IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies** — Hosseinkhani, Peng, Shramko et al. — [arXiv:2609.10915](https://arxiv.org/abs/2609.10915) — *Talk Mon 9:00, Room 329 · **Poster #4624** (session M1FA09): board D8 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Replaces iterative diffusion/flow action heads with a single-step cIMLE generator for fast VLA action generation.
- ✅ **StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models** — Pangaonkar, Rath, Patil et al. — [arXiv:2603.20659](https://arxiv.org/abs/2603.20659) — *Talk Mon 15:27, Room 334 · **Poster #3412** (session M2FB11): board G9 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Training-free VLM reasoning over rollout videos to rearrange initial scene state, mitigating VLA distractor and obstruction failures.

## P2 — Nice to Read

### VLA Models
- **Embodied Chain-Of-Thought Model Via Interleaved Reasoning** — Liu, Zhao, Zhang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Embodied%20Chain-Of-Thought%20Model%20Via%20Interleaved%20Reasoning%22) — *Talk Wed 15:54, Room 409/410 · **Poster #1724** (session W2LBY07): board C5 (South face), Exhibit Hall B, Wed 16:30–18:00*  
  Embodied CoT VLA that interleaves subgoal image prediction and textual reasoning in one pass using modality-specialized MoE experts.
- ✅ **DualCoT-VLA: Visual-Linguistic Chain of Thought Via Parallel Reasoning for Vision-Language-Action Models** — Zhong, Li, He et al. — [arXiv:2603.22280](https://arxiv.org/abs/2603.22280) — *Talk Tue 14:30, Room 317/318 · **Poster #2500** (session T2LAX04): board D9 (North face), Exhibit Hall B, Tue 16:30–18:00*  
  VLA with parallel visual and linguistic chain-of-thought via learnable queries, avoiding autoregressive reasoning latency.
- ✅ **Improving Vision-Language-Action Model Fine-Tuning with Structured Stage and Keyframe Supervision** — Xu, Chen, Wang et al. — [arXiv:2606.26801](https://arxiv.org/abs/2606.26801) — *Talk Wed 14:43, Room 411/412 · **Poster #2610** (session W2LAX08): board E10 (North face), Exhibit Hall B, Wed 16:30–18:00*  
  Auxiliary stage-classification and next-gripper-keyframe heads derived automatically from demos improve VLA fine-tuning on long-horizon tasks.
- ✅ **OG-VLA: Orthographic Image Generation for 3D-Aware Vision-Language Action Model** — Singh, Goyal, Birchfield et al. — [arXiv:2506.01196](https://arxiv.org/abs/2506.01196) — *Talk Mon 10:18, Room 401/402 · **Poster #3972** (session M1LBY05): board E3 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  3D-aware VLA rendering canonical orthographic views and generating end-effector target images; strong generalization on Arnold/Colosseum.
- ✅ **3D HAMSTER: Bridging Planning and Control in Hierarchical Vision Language Action Models through 3D Trajectory Guidance** — Hwang, Lee, Kim et al. — [arXiv:2606.31329](https://arxiv.org/abs/2606.31329) — *Talk Tue 15:25, Room 317/318 · **Poster #4393** (session T2LBX04): board E8 (West face), Exhibit Hall B, Tue 16:30–18:00*  
  Hierarchical VLA where the VLM predicts 3D rather than 2D end-effector trajectories to guide point-cloud low-level policies.
- **Hand-Centric Interaction Representation for Vision-Language-Action Models** — Kaichi, Kambara, Wang et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Hand-Centric%20Interaction%20Representation%20for%20Vision-Language-Action%20Models%22) — *Talk Wed 14:46, Room 411/412 · **Poster #4642** (session W2LAX08): board E10 (South face), Exhibit Hall B, Wed 16:30–18:00*  
  Hand-centric flow-matching action representation from human video (pose plus contact) for VLA pretraining.
- ✅ **TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation** — Zhang, Zhang, Xu et al. — [arXiv:2603.12665](https://arxiv.org/abs/2603.12665) — *Talk Mon 15:04, Room 304/305 · **Poster #3006** (session M2LAY02): board B4 (North face), Exhibit Hall B, Mon 16:30–18:00*  
  Contact-gated tactile tokens fused into a fine-tuned VLA for contact-rich manipulation.

### RL Fine-Tuning & Policy Optimization
- **StageNav: A Diagnostic Benchmark and State-Aligned RL Framework for Long-Horizon Navigation** — Fang, Yang, Liu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22StageNav%3A%20A%20Diagnostic%20Benchmark%20and%20State-Aligned%20RL%20Framework%20for%20Long-Horizon%20Navigation%22) — *Talk Wed 10:09, Room 409/410 · **Poster #1244** (session W1LBY07): board D11 (South face), Exhibit Hall B, Wed 11:00–12:30*  
  Reinforcement fine-tuning of LVLM navigators with dense process rewards from verified subgoal-state predictions, plus diagnostic long-horizon benchmark.
- ✅ **JOP-VLN: Joint On-and-Off Policy Learning for Vision-and-Language Navigation** — He, Zhao, Zheng et al. — [arXiv:2607.13461](https://arxiv.org/abs/2607.13461) — *Talk Wed 15:11, Room 401/402 · **Poster #1230** (session W2LAY05): board F8 (North face), Exhibit Hall B, Wed 16:30–18:00*  
  Three-stage VLN training combining IL, DAgger and joint on/off-policy RL with high-entropy sampling, SOTA on R2R.

### Robot Learning & Imitation
- ✅ **CubeDAgger: Interactive Imitation Learning for Dynamic Systems with Efficient yet Low-Risk Interaction** — Kobayashi — [arXiv:2505.04897](https://arxiv.org/abs/2505.04897) — *Talk Tue 10:09, Room 403/404 · **Poster #1590** (session T1LBY06): board A10 (North face), Exhibit Hall B, Tue 11:00–12:30*  
  Interactive imitation learning for dynamic tasks using consensus of action candidates and colored-noise exploration to keep stability.
- **Energy-Score Stabilized Attentive Mixture-Density Policy: A Lightweight Approach to Multimodal Robot Motion** — Fujita, Ichiwara, Sugano et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Energy-Score%20Stabilized%20Attentive%20Mixture-Density%20Policy%3A%20A%20Lightweight%20Approach%20to%20Multimodal%20Robot%20Motion%22) — *Talk Tue 14:30, Room 320 · **Poster #2389** (session T2FA02): board G7 (South face), Exhibit Hall B, Tue 16:30–18:00*  
  Mixture-density policy trained with Energy Score instead of NLL gives single-pass multimodal actions, beating Diffusion Policy in small-data regimes.
- ✅ **From Code to Action: Hierarchical Learning of Diffusion-VLM Policies** — Peschl, Mazzaglia, Dijkman — [arXiv:2509.24917](https://arxiv.org/abs/2509.24917) — *Talk Mon 10:27, Room 401/402 · **Poster #2158** (session M1LBY05): board E3 (West face), Exhibit Hall B, Mon 11:00–12:30*  
  Uses robot API subtask functions as supervision: a code-generating VLM decomposes tasks grounded by diffusion policies with memory.

### World Models
- **Physics-Informed Policy Learning for Floating-Base Robots** — Yang, Liu, Ding et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Physics-Informed%20Policy%20Learning%20for%20Floating-Base%20Robots%22) — *Talk Wed 9:40, Room 304/305 · **Poster #1830** (session W1LAY02): board A5 (North face), Exhibit Hall B, Wed 11:00–12:30*  
  Grey-box Lagrangian dynamics model with learned contact forces as a drop-in for MBPO, yielding more reliable imagined rollouts for locomotion.
- ✅ **NavThinker: Action-Conditioned World Model for Coupled Prediction and Planning in Social Navigation** — HU, Gong, Kong et al. — [arXiv:2603.15359](https://arxiv.org/abs/2603.15359) — *Talk Mon 14:33, Room 403/404 · **Poster #3700** (session M2LAX06): board C9 (East face), Exhibit Hall B, Mon 16:30–18:00*  
  Action-conditioned world model in Depth-Anything feature space feeds think-ahead features and reward shaping to PPO for social navigation.

### Generalization, Sim-to-Real & Evaluation
- **RobustFuzz: A Semantic Disturbance Benchmark for Learned Policies in Embodied Control** — Ren — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22RobustFuzz%3A%20A%20Semantic%20Disturbance%20Benchmark%20for%20Learned%20Policies%20in%20Embodied%20Control%22) — *Talk Mon 10:04, Room 411/412 · **Poster #1397** (session M1LBX08): board F10 (East face), Exhibit Hall B, Mon 11:00–12:30*  
  Fuzzing benchmark that searches for timing/hardware disturbance schedules that break learned control policies.
- **What Good Is a Good Robot Model? Differentiable Regression, Parameter Ablation, and Policy Performance of Quadrupedal Models** — Hackett, Hubicki — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22What%20Good%20Is%20a%20Good%20Robot%20Model%3F%20Differentiable%20Regression%2C%20Parameter%20Ablation%2C%20and%20Policy%20Performance%20of%20Quadrupedal%20Models%22) — *Talk Wed 15:54, Room 304/305 · **Poster #3620** (session W2LBY02): board C8 (West face), Exhibit Hall B, Wed 16:30–18:00*  
  Systematically compares fine-tuned parametric models vs domain randomization for quadruped sim-to-real policy performance.

### RL Algorithms & Exploration
- ✅ **Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning** — Shin, Ren, Xiong et al. — [arXiv:2603.19078](https://arxiv.org/abs/2603.19078) — *Talk Wed 15:57, Room 409/410 · **Poster #947** (session W2LBY07): board C5 (West face), Exhibit Hall B, Wed 16:30–18:00*  
  Policy network architecture built on Articulated Body Algorithm inertia propagation improves RL sample efficiency and dynamics-shift robustness.
- ✅ **MO-Playground: Massively Parallelized Multi-Objective Reinforcement Learning for Robotics** — Janwani, Novoseller, Lawhern et al. — [arXiv:2603.09237](https://arxiv.org/abs/2603.09237) · [RA-L](https://doi.org/10.1109/lra.2026.3700381) — *Talk Tue 14:48, Room 409/410 · **Poster #5022** (session T2LAX07): board E7 (East face), Exhibit Hall B, Tue 16:30–18:00*  
  GPU-native multi-objective RL algorithm and parallel environment suite for Pareto policy families in robotics.
- **Climb with SHERPA: Heuristic-Guided Reinforcement Learning Via Segmented Experience Relay** — Kim, Lee, Lee et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Climb%20with%20SHERPA%3A%20Heuristic-Guided%20Reinforcement%20Learning%20Via%20Segmented%20Experience%20Relay%22) — *Talk Wed 15:38, Room 329 · **Poster #4955** (session W2FB09): board B8 (North face), Exhibit Hall B, Wed 16:30–18:00*  
  Alternates heuristic and RL control in contiguous segments to guide exploration in sparse-reward long-horizon manipulation.
- ✅ **EFGCL: Dynamic Motion Acquisition through Spotting-Inspired External Force Guided Curriculum Learning** — Yoneda, Kawaharazuka, Okada — [arXiv:2605.10063](https://arxiv.org/abs/2605.10063) — *Talk Tue 9:13, Room 317/318 · **Poster #4756** (session T1LAX04): board B11 (South face), Exhibit Hall B, Tue 11:00–12:30*  
  External assistive forces during training as physical-guidance curriculum let RL learn backflips and jumps faster.
- ✅ **WARL: Wrench-Augmented Reinforcement Learning for Task-Agnostic Learning in Legged Robots** — Yoneda, Kawaharazuka, Okada — [arXiv:2607.24036](https://arxiv.org/abs/2607.24036) — *Talk Wed 15:04, Room 304/305 · **Poster #2509** (session W2LAY02): board D8 (North face), Exhibit Hall B, Wed 16:30–18:00*  
  Adds wrench actions for exploration in legged RL, annealed by curriculum; analyzes when it hurts embodiment use.

### Efficient VLA Inference
- ✅ **TIDAL: Temporally Interleaved Diffusion and Action Loop for Dynamic Manipulation** — Sun, Wang, Bai et al. — [arXiv:2601.14945](https://arxiv.org/abs/2601.14945) — *Talk Wed 15:16, Room 411/412 · **Poster #1942** (session W2LBX08): board D10 (South face), Exhibit Hall B, Wed 16:30–18:00*  
  Dual-frequency scheduler interleaving single-step flow integration with execution and staleness-aware training for dynamic VLA manipulation.
- ✅ **Shallow-π: Knowledge Distillation for Flow-Based VLAs** (Award Candidate) — Jeon, Choi, Kim — [arXiv:2601.20262](https://arxiv.org/abs/2601.20262) — *Talk Mon 14:30, Room 406 · **Poster #85** (session M2AA01): board B7 (West face), Exhibit Hall B, Mon 16:30–18:00*  
  Distills flow-based VLA (pi0-style) from 18 to 6 transformer layers, 2x faster inference with <1% success drop on Jetson.

### Safe RL
- ✅ **Directional Constraints for Efficient Exploration in Safe Reinforcement Learning** — Magliano, Liu, Peters et al. — [arXiv:2607.12784](https://arxiv.org/abs/2607.12784) — *Talk Tue 15:16, Room 320 · **Poster #3012** (session T2FB02): board G6 (South face), Exhibit Hall B, Tue 16:30–18:00*  
  Extends the ATACOM safety layer with directional constraints that only activate when moving toward constraint boundaries, improving safety-performance trade-off.
- **Safe Offline Reinforcement Learning via Chance-Constrained Policy Filtering** — Kim, Yoo, Ahn — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Safe%20Offline%20Reinforcement%20Learning%20via%20Chance-Constrained%20Policy%20Filtering%22) — *Talk Mon 9:00, Room 320 · **Poster #3458** (session M1FA02): board F2 (North face), Exhibit Hall B, Mon 11:00–12:30*  
  Chance-constrained policy filter with posterior-calibrated cost estimator for safe offline RL with adjustable thresholds at deployment.

### Additional
- **Robotic Long-Horizon Manipulation with Bayesian Non-Parametric Skill Priors** — Meng, Yao, Wu et al. — [IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22Robotic%20Long-Horizon%20Manipulation%20with%20Bayesian%20Non-Parametric%20Skill%20Priors%22) — *Talk Mon 10:09, Room 411/412 · **Poster #263** (session M1LBY08): board F10 (South face), Exhibit Hall B, Mon 11:00–12:30*  
  Dirichlet-process mixture skill prior for hierarchical RL improves exploration on sparse-reward long-horizon manipulation vs. single-Gaussian priors.
- ✅ **Demonstration-Free Robotic Control Via LLM Agents** — Tsui, Fang, Hwu — [arXiv:2601.20334](https://arxiv.org/abs/2601.20334) — *Talk Tue 9:33, Room 328 · **Poster #1480** (session T1FA08): board B2 (West face), Exhibit Hall B, Tue 11:00–12:30*  
  Runs an unmodified frontier LLM coding agent as a demonstration-free manipulation controller, approaching few-shot VLA success on LIBERO/ManiSkill/MetaWorld.
- ✅ **LooperMuscle: Fast and Stable Learning of Humanoid Whole-Body Tracking Via Structured Mixture-Of-Experts** — Liu, Li, Yu et al. — [arXiv:2608.00820](https://arxiv.org/abs/2608.00820) — *Talk Mon 15:11, Room 401/402 · **Poster #4631** (session M2LAY05): board D13 (North face), Exhibit Hall B, Mon 16:30–18:00*  
  Structured mixture-of-experts actor, distributional critic and routed replay close gap between FastSAC and PPO for humanoid tracking in 45 min.
- ✅ **CORAL: Scalable Multi-Task Robot Learning Via LoRA Experts** — Luo, Chen, Liang et al. — [arXiv:2603.09298](https://arxiv.org/abs/2603.09298) — *Talk Mon 9:18, Room 411/412 · **Poster #2281** (session M1LAX08): board G11 (West face), Exhibit Hall B, Mon 11:00–12:30*  
  Frozen VLA backbone with swappable per-task LoRA experts avoids negative transfer in multi-task deployment.
- ✅ **VR-DAgger: Immersive VR for Dexterous Data Collection and Uncertainty-Guided On-Policy Correction** — Zurbrügg, Portela, Bhardwaj et al. — [arXiv:2605.27114](https://arxiv.org/abs/2605.27114) — *Talk Wed 9:04, Room 302/303 · **Poster #3260** (session W1LAX01): board D4 (East face), Exhibit Hall B, Wed 11:00–12:30*  
  VR teleoperation with MC-dropout uncertainty selecting failure segments for targeted DAgger-style corrections of dexterous diffusion policies.

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
- I started from all 1,929 papers in the official Paper & Author Index.
- I kept 742 candidates: the 719 whose title or abstract mentions a core learning topic, plus 23 that an earlier title-only pass had flagged or that have no abstract to check.
- Each candidate was scored against my group's research areas from its full abstract, with an LLM doing the first pass. I then reviewed the result and trimmed and adjusted the list by hand.

arXiv preprints (✅) were matched by title where available. Some papers without a preprint may have one posted after the conference. If you're at IROS 2026 in Pittsburgh and this overlaps with your interests, I hope it saves you some filtering time.

**Sources:**
- [IROS 2026 Paper & Author Index](https://2026.ieee-iros.org/program/paper-index/): the official program, and the source of the titles, authors, abstracts, talk times, rooms, session codes and poster sessions. Every paper on this list was matched to it by paper number.
- [IROS 2026 poster map](https://2026.ieee-iros.org/program/poster-sessions/): the board numbers and faces in Exhibit Hall B.
- [IROS 2026 Paper Explorer](https://gisbi-kim.github.io/iros2026-explorer/output/iros2026_explorer.html) by Giseop Kim: the starting point for the full list of papers, authors and keywords.

*A note on selection: this list is based on my own research interests, not on the quality of the work. There are great papers across all of IROS 2026, and many excellent ones aren't here simply because they fall outside my group's areas.*

{% comment %}
SOCIAL MEDIA DRAFTS (Liquid comment: stripped from the built page)
URL: https://www.fracturedplane.com/blog/2026/09/28/IROS2026-Reading-List.html

--- twitter ---
IROS 2026 has 1,929 papers. I filtered them down to a prioritized reading list of 142 on robot learning, VLAs, RL fine-tuning of robot policies, imitation learning, and world models, with session times, rooms, and the related workshops. {{URL}}

--- bluesky ---
I filtered the 1,929 IROS 2026 papers down to a prioritized reading list on robot learning, VLAs, RL fine-tuning, and world models, with session times and rooms. {{URL}}

--- linkedin ---
IROS 2026 in Pittsburgh has 1,929 papers in its program. Nobody is reading all of them, so I filtered the list down to what's relevant to my group's research. I'm sharing it in case it's useful to others at the conference.

Rather than filtering on titles alone, I used the abstracts the IROS program site publishes for almost every paper. The result is 142 papers grouped into must read, should read, and nice to read. Each entry has a one-line summary, an arXiv link where there is a preprint, and its session time and room.

A few threads worth following this year:
• RL fine-tuning of VLAs, the robotics side of the RLVR/GRPO work on LLMs
• Careful empirical studies of what makes imitation learning succeed or fail
• Evaluating how well VLAs generalize
• World models for policy learning, data synthesis, and evaluation

The post also lists the workshops closest to robot learning, VLAs, world models, and sim-to-real, including the ones still to come on Thursday.

{{URL}}

#IROS2026 #RobotLearning #ReinforcementLearning #VLA #WorldModels
{% endcomment %}
