---
title: "Knowledge vs Information, Revisited"
date: 2026-09-28
description: "An old note from grad school on the difference between knowledge and information, and why foundational models leave us with more research questions, not fewer."
summary: "Many students feel there is little research left to do now that foundational models can accomplish so much. I believe the opposite is true: foundational models are an existence proof that a capable policy can be built, and that existence proof raises deep questions about how to reproduce foundational models, make foundational models more efficient, and understand how foundational models acquire their knowledge."
cover: <img width="70%" src="/assets/projects/knowledge-vs-information.svg">
category: Article
tags:
   - research
   - foundation-models
   - grad-school
author: Glen Berseth
authors: Glen Berseth
draft: false
layout: page
type: Article
titleShort: Knowledge vs Information
---

I wrote the first version of this post in 2015 as a graduate student, and the draft has remained unpublished since then. I am revisiting the draft now because of recent conversations with students, many of whom shared the feeling that little research remains to be done because foundational models can already accomplish so much. I find the opposite view, that now there are many more unanswered questions more compelling, and this old figure offers a useful way to explain why.

## The original idea

Having spent a considerable amount of time studying and improving machine learning models, I believe there is an important difference between *knowledge* and *information*. The internet is full of information: countless small, *independent pieces of data*. In research, students spend most of their time learning *rules* that can be reused. *Learning* then becomes the process of connecting pieces of information together to build knowlege.

<div align="center"><img src="/assets/projects/knowledge-vs-information.svg" alt="Knowledge vs Information: Bewilderment, Confusion, Ambiguity, and Wisdom" width="60%"></div>

With little knowledge and little information, we are left in *bewilderment*. A large amount of information without the knowledge to organize that information leads to *ambiguity*. Knowledge without enough information to apply that knowledge leaves us with *confusion* to if our rules apply to other situations. Only the combination of knowledge and information produces *wisdom*. As a graduate student, I generated a large amount of information and rarely had enough time to study all of that information.

## Where foundational models fit

Language models and other foundational models are an existence proof of a policy that can perform a particular task. Previously, we did not know whether a single model could write code, answer questions across many subjects, or follow instructions in natural language. Now we know that such a model is possible. Foundational models represent a large step along the information axis, but foundational models do not, by themselves, move us along the knowledge axis. We have the result, with far less understanding of why the result works.

This gap is where I see a large number of open research questions:
1. How can we reproduce foundational models, and which parts of the training recipe actually matter?
2. How can we make foundational models faster and cheaper to train and deploy?
3. How do foundational models acquire the knowledge they have, and how is that knowledge represented?
4. How can foundational models be improved, rather than only scaled?

In many ways, I expect the introduction of foundational models to help focus our scientific direction. An existence proof tells researchers where to look. An existence proof also gives researchers a concrete object to analyze, along with many deep questions about how foundational models are built and how foundational models can be improved. Rather than leaving less research to do, foundational models have given us a great deal of information, and the work of turning that information into knowledge is just beginning.
