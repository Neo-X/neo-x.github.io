---
title: "Where do Foundational Models Fit Into Robotics Progress"
date: 2026-09-15
description: "A short reflection on how recent foundational models fit into decades of robotics progress, and what still needs to improve for flexible automation."
summary: "Foundational models have made a decades-old class of manipulation problems dramatically cheaper to solve, but cheaper is not the same as more capable. This post frames recent progress against a performance/cost trade-off curve and lays out what it will take to push the ceiling on what automation can do, not just the cost of what it already can."
cover: <img width="100%" src="/assets/projects/foundational-models-robotics/performance-cost-tradeoff-future.svg">
category: Article
tags:
   - robotics
   - foundation-models
   - manipulation
   - automation
author: Glen Berseth
authors: Glen Berseth
draft: false
layout: page
type: Article
titleShort: Where do Foundational Models Fit Into Robotics
---

I'm writing this short article to help put into perspective some of the recent progress of using larger foundational models on robotics tasks, to better frame their success and where progress is still needed [[1](https://x.com/chooi_jeq/status/2096064315115839904?s=46), [2](https://anonymous-report-421.github.io/public-website/?lang=en&view=1)].

Robotics is not a new field. Here we show an example of some of the [early work in Minsky's lab](https://infinite.mit.edu/video/eye-robot-studies-machine-vision-mit-and-tx-o-computer-1959/) being able to use computer vision to understand the geometry of the objects in the scene, and plan a path for the robot arm to pick up objects and assemble a defined collection of objects.

<div align="center">
<video width="75%" controls>
    <source src="/assets/projects/foundational-models-robotics/minsky-lab-demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
</div>

The recent foundational models have improved further on these types of problems, of manipulating objects in a fixed scene, and in order to be able to do such a thing with reduced time needed by the developer to produce those solutions. However, it really begins to start to ask the question, what direction should we be taking the robotics field in order to make the performance improvements needed for the future of flexible automation? In most engineering fields, a goal of the field is to better understand the trade-offs between the performance and cost curve for a set of possible solutions, where cost here means some combination of time, money, and energy.

<div align="center"><img src="/assets/projects/foundational-models-robotics/performance-cost-tradeoff-original.svg" alt="Performance vs. cost trade-off curve" width="75%"></div>

Given that we've been able to tackle similar tasks for many decades, I would say that we've been able to get fairly high performance on similar tasks of manipulating types of objects and moving them around in fixed environments. However, the space of possible situations that we can automate is much larger than just these tasks. For example, for trying to be able to produce self-driving cars, the performance has been a much slower improvement for many years. Much of this challenge is due to the constraint of needing to perform closed loop feedback in order to adjust quickly and intelligently to changes in the environment around the robot. As Jitendra Malik [recently said](https://x.com/JitendraMalikCV/status/2097173961264284039), "robotics is also about dexterity and dynamics". These more dexterous and dynamic tasks are areas foundational models perform poorly.

<div align="center"><img src="/assets/projects/foundational-models-robotics/performance-cost-tradeoff-future.svg" alt="Performance vs. cost trade-off curve, showing the goal of pushing performance higher, not just cost lower" width="75%"></div>

Shifting this curve up means making improvements on:
1. Larger object variety
2. Dealing with stochasticity to enable working in the wild
3. Progress on multi-robot and human robot interaction and planning
4. Realtime planning and control
5. Better hardware, human hands are impressive

None of this diminishes what foundational models have already done: they've made a decades-old class of problems dramatically cheaper to solve, and that is valuable. But cheaper is not the same as more capable. The real measure of progress for this field won't be how far left we've pushed the cost of automating what a 1960s lab could already do in principle, it'll be how far up we've pushed the ceiling on what automation can accomplish.
