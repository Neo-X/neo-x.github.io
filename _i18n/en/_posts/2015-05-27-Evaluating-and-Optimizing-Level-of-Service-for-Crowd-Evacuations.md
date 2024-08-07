---
title: Evaluating and Optimizing Level of Service for Crowd Evacuations
date: 2015-05-20 10:20
modified: Sunday, 21. March 2017 02:06PM 
category: Publication
Tags: CrowdSimulation, CrowdEvacuation, SteerSim, Optimization
author: Glen B
authors: Brandon Haworth, Muhammad Usman, Glen Berseth, Mubbasir Turab Kapadia, Petros Faloutsos
cover: <div> <img width="100%" src="/assets/projects/OptLOS/teaser.png"/> </div>
summary: Computational approaches for crowd analysis and environment design need robust measures for characterizing the relation between environments and crowd flow. Level of service (Level of Service) is a standard indicator for characterizing the service afforded by environments to crowds of specific densities, and is widely used in crowd management and urban design. However, current Level of Service indicators are qualitative and rely on expert analysis. In this paper, we perform a systematic analysis of Level of Service for synthetic crowds. The flow-density relationships in crowd evacuation scenarios are explored with respect to three state-of-the-art steering algorithms. Our results reveal that Level of Service is sensitive to the crowd behavior, and the configuration of the environment benchmark. Following this study, we automatically optimize environment elements to maximize crowd flow, under a range of Level of Service conditions. The steering algorithm, the number of optimized environment elements, the scenario configuration and the Level of Service conditions affect the optimal configuration of environment elements. We observe that the critical density of crowd simulators increases due to the optimal placement of pillars, thereby effectively increasing the Level of Service of environments at higher crowd densities. Depending on the simulation technique and environment benchmark, pillars are configured to produce lanes or form wall-like structures, in an effort to maximize crowd flow. These experiments serve as an important precursor to computational crowd optimization and management and motivate the need for further study using additional real and synthetic crowd datasets across a larger representation of environment benchmarks.
Type: EnvironmentDesign
TitleShort: Automated environment design for level-of-service
---

In this work we study the effects environment optimization has on the level of service.

<div> <img width="100%" src="/assets/projects/OptLOS/teaser.png"/> </div>

## Abstract

Computational approaches for crowd analysis and environment design need robust measures for characterizing the relation between environments and crowd flow. Level of service (Level of Service) is a standard indicator for characterizing the service afforded by environments to crowds of specific densities, and is widely used in crowd management and urban design. However, current Level of Service indicators are qualitative and rely on expert analysis. In this paper, we perform a systematic analysis of Level of Service for synthetic crowds. The flow-density relationships in crowd evacuation scenarios are explored with respect to three state-of-the-art steering algorithms. Our results reveal that Level of Service is sensitive to the crowd behavior, and the configuration of the environment benchmark. Following this study, we automatically optimize environment elements to maximize crowd flow, under a range of Level of Service conditions. The steering algorithm, the number of optimized environment elements, the scenario configuration and the Level of Service conditions affect the optimal configuration of environment elements. We observe that the critical density of crowd simulators increases due to the optimal placement of pillars, thereby effectively increasing the Level of Service of environments at higher crowd densities. Depending on the simulation technique and environment benchmark, pillars are configured to produce lanes or form wall-like structures, in an effort to maximize crowd flow. These experiments serve as an important precursor to computational crowd optimization and management and motivate the need for further study using additional real and synthetic crowd datasets across a larger representation of environment benchmarks.

## Files


[Paper](/assets/projects/OptLOS/MIG_2015_LOS.pdf)
[comment]: <> ([Presentation](/assets/projects/GameLevelOptimization/paper_errata.pdf))
[comment]: <> ( [Code](https://github.com/FracturedPlane/EnvironmentInterface))

```
@inproceedings{Haworth:2015:EOL:2822013.2822040,
 author = {Haworth, Brandon and Usman, Muhammad and Berseth, Glen and Kapadia, Mubbasir and Faloutsos, Petros},
 title = {Evaluating and Optimizing Level of Service for Crowd Evacuations},
 booktitle = {Proceedings of the 8th ACM SIGGRAPH Conference on Motion in Games},
 series = {MIG '15},
 year = {2015},
 isbn = {978-1-4503-3991-9},
 location = {Paris, France},
 pages = {91--96},
 numpages = {6},
 url = {http://doi.acm.org/10.1145/2822013.2822040},
 doi = {10.1145/2822013.2822040},
 acmid = {2822040},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {crowd evacuation, crowd simulation, environment optimization, level of service, steering algorithms},
} 
```
