---
title: "The First Two Chapters of a Robot Learning Book"
date: 2026-10-25
description: "Draft chapters on what robot learning is and on imitation learning are now online at robotlearningbook.com, built from three years of my Robot Learning course."
summary: "I am writing an open textbook on robot learning, Foundations of Robot Learning: Scaling Learning for Real-World Agents. The first two chapters, on what robot learning is and on imitation learning, are now online as drafts. The book is built from the slides, lecture notes and recordings of my Robot Learning course at Université de Montréal and Mila over the past three years, and its goal is to give people learning robotics better source material than slides and scattered papers."
category: Article
tags:
   - robot-learning
   - teaching
   - book
   - imitation-learning
author: Glen Berseth
authors: Glen Berseth
draft: false
layout: page
type: Article
titleShort: Robot Learning Book
cover: <img width="70%" src="/assets/projects/robot-learning-book/morphology-task-grid.svg" alt="The tasks-by-morphologies grid. Rows are robot bodies (aerial, quadruped, humanoid), columns are tasks (clean, assemble, search). Data covers single cells, and a generalist robot policy should perform well across the full grid.">
social:
   linkedin: https://www.linkedin.com/feed/update/urn:li:share:7520173421649895424/
   bluesky: https://bsky.app/profile/glenberseth.bsky.social/post/3myppxx3jyy22
---

I am writing an open textbook on robot learning, *Foundations of Robot Learning: Scaling Learning for Real-World Agents*. The first two chapters are now online as drafts at [robotlearningbook.com](https://robotlearningbook.com/), each next to the slides and lecture recordings for the same topic.

## Why write a book

Robot learning moves quickly, and the material for learning it has not kept up. A student who wants to enter the field today pieces it together from lecture slides, recorded talks, blog posts and a growing pile of papers, each written for a different audience and with its own notation. Slides are terse by design, and papers assume the reader already knows the twenty papers that came before. There are good textbooks on reinforcement learning and on classical robotics, but few that treat learning *for robots* as one subject, from imitation learning through deep reinforcement learning to large foundation models.

I want to provide better source material for people learning robotics: one place, in one consistent notation, that explains not only what each method does but why it was needed and where it still fails. The book is free to read for students at any institution and for practitioners who are not enrolled anywhere.

## What is in the first two chapters

<img src="/assets/projects/robot-learning-book/morphology-task-grid.svg" alt="The tasks-by-morphologies grid: rows represent robot embodiments, columns represent tasks. An ideal generalist robot policy achieves high performance across the full grid." style="float: right; width: 40%; margin: 0 0 1em 1.5em;">
**Chapter 1: What Is Robot Learning?** The first chapter starts with an honest look at where robots stand today. Robots are already common in manufacturing, warehouses and hospitals, but they work there because the environment is engineered to tolerances that make adaptability unnecessary. The chapter argues that learning is what closes the gap between these brittle, pre-scripted systems and the robots we actually envision. A useful way to picture the goal is a grid of tasks by robot bodies, where each row is a robot such as a humanoid, a quadruped or a drone, and each column is a task such as cleaning, assembly or search. Today, most data and most policies cover single cells of this grid, and transferring a skill from one cell to another is hard. A generalist robot policy should instead perform well across the whole grid.

The chapter then sets up the notation used throughout the book (states versus observations, the Markov property, policies and trajectories), and surveys the three main paradigms: learning from demonstrations, autonomous learning with deep reinforcement learning, and model-based methods. It closes on the questions that motivate the rest of the book: how to find representations that generalize, where the data comes from, and whether reinforcement learning can be fed enough experience to scale.

<div style="clear: both;"></div>

<img src="/assets/projects/robot-learning-book/tightRope.svg" alt="Tight rope walking: at each step the policy slips with probability epsilon, and a slip takes it out of the training distribution." style="float: right; width: 45%; margin: 0 0 1em 1.5em;">
**Chapter 2: Supervised Learning and Imitation Learning.** The second chapter asks whether a simple recipe, observing what experts do and copying them, can serve as a foundation for general robot learning. It traces imitation learning from ALVINN in 1988, through large-scale systems such as BC-Z and RT-1, to today's vision-language-action (VLA) models, which build on pretrained vision-language models and are trained largely by imitation learning on large collections of robot demonstrations. The chapter also explains why the simple recipe breaks. The simplest form of imitation learning, behaviour cloning, is like walking a tightrope: at each step the policy slips with some small probability, and a single slip takes it into states the expert never visited, where the training data offers no guidance. Staying on the rope for a whole task becomes less likely the longer the task, and the total cost grows quadratically with the task horizon.

From there, the chapter covers the main fixes and why each was needed: DAgger for distribution shift, scale and data diversity for spurious correlations, diffusion, variational and autoregressive models for multi-modal demonstrations, and action chunking to shorten the effective horizon. Action chunking also matters for VLAs, whose large models can take half a second or more per prediction, too slow for a control loop running at 50 Hz unless the model predicts many actions at once. The chapter finishes with learning from demonstrations without action labels, inverse dynamics models, inverse reinforcement learning and adversarial imitation.

<div style="clear: both;"></div>

## How the book is made

The content comes from my Robot Learning course (IFT 6163) at Université de Montréal and Mila, and in particular from the slides, lecture notes and recordings of the past three years. The slides fix the order of topics, the figures and the mathematics. The recordings capture what the slides leave out: the intuition behind an equation, the reason one method replaced another, and the examples that only come out when answering students' questions. I use AI tools to turn many hours of transcribed lectures into first drafts that follow the slides, and then I read, correct and revise every chapter myself. Because the book is built from the same material I update each time I teach, improvements to the course flow back into the book. The [How This Book Is Made](https://robotlearningbook.com/about-this-book/) page has more detail.

## Read it and send feedback

These are drafts, and the remaining chapters on reinforcement learning, planning, goal-conditioned learning and foundation models for robotics will follow as they are ready. You can read [Chapter 1](https://robotlearningbook.com/lectures/lec00/) and [Chapter 2](https://robotlearningbook.com/lectures/lec01/) online or download them as PDFs. If you find a mistake, an unclear explanation or a missing reference, please let me know. Every correction improves both the book and the next offering of the course.

{% comment %}
SOCIAL MEDIA DRAFTS (Liquid comment: stripped from the built page)
URL: https://www.fracturedplane.com/blog/2026/10/25/robot-learning-book-first-chapters.html
X and Bluesky post as threads: the cover image is attached to the first post, the reply is threaded under it.
X is posted by hand.

--- twitter ---
I am writing an open textbook on robot learning; the first two draft chapters are online. Chapter 1: what robot learning is and why today's robots are brittle. Chapter 2: imitation learning, from ALVINN to today's VLA models, and why behaviour cloning is like walking a tightrope.

--- twitter reply ---
The book is built from three years of slides, notes and lecture recordings from my Robot Learning course at UdeM and Mila, to give people learning robotics better source material.

Book: https://robotlearningbook.com/
How it is made: {{URL}}

--- bluesky ---
I am writing an open textbook on robot learning, and the first two draft chapters are online. Chapter 1 asks what robot learning is and why today's robots are brittle. Chapter 2 covers imitation learning, from ALVINN to today's VLA models, and why behaviour cloning is like walking a tightrope.

--- bluesky reply ---
Built from three years of my Robot Learning course at UdeM and Mila, to give people learning robotics better source material.

Book: https://robotlearningbook.com/
How it is made: {{URL}}

--- linkedin ---
Robot learning moves quickly, and the material for learning it has not kept up. Students piece the field together from slides, recorded talks and a growing pile of papers, each with its own notation. I want to provide better source material for people learning robotics.

So I am writing an open textbook, Foundations of Robot Learning: Scaling Learning for Real-World Agents. The first two chapters are now online as drafts:
• What Is Robot Learning? Why today's robots are brittle, the notation used throughout the book, and the main paradigms: imitation learning, deep reinforcement learning and model-based methods. The figure shows one way to picture the goal: a grid of robot bodies by tasks, where today's data covers single cells and a generalist robot policy should perform well across the whole grid.
• Supervised Learning and Imitation Learning: from ALVINN to BC-Z, RT-1 and today's vision-language-action (VLA) models, why behaviour cloning is like walking a tightrope, and the fixes, from DAgger to diffusion models and action chunking.

The book is built from the slides, lecture notes and recordings of my Robot Learning course (IFT 6163) at Université de Montréal and Mila over the past three years. I use AI tools to turn the transcribed lectures into first drafts, and then I read, correct and revise every chapter myself.

The chapters are free to read. If you find a mistake or an unclear explanation, please let me know.

Read the book: https://robotlearningbook.com/
How the book is made: {{URL}}

#RobotLearning #Robotics #ImitationLearning #MachineLearning #OpenEducation
{% endcomment %}
