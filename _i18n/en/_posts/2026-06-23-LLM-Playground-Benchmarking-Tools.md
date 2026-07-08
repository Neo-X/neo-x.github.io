---
title: "Taking Control of My Data: Why I Started Building My Own LLM Tools"
date: 2026-06-23
description: "The real motivation behind llm-playground wasn't benchmarking for its own sake — it's about owning my data and automating the task tracking, meeting notes, and documentation work I used to hand off to other tools. Benchmarking was just step one: picking a local model that's actually good enough to trust with that data."
summary: "I started building llm-playground to take control of my own data and stop depending on third-party tools for task tracking, meeting notes, and documentation. Before any of that could work, I needed to know which local LLMs I could actually run well on my own compute, and that speed alone isn't the right thing to optimize for — accuracy matters more once you're trusting a model with your own workflow. This post covers the benchmarking foundation; dedicated posts on the task-tracking, meeting-transcript, and documentation tools are coming."
category: Tools
tags:
   - LLMs
   - benchmarking
   - local-inference
   - ollama
   - llama.cpp
   - tooling
   - data-ownership
draft: false
layout: page
type: Professional Development
titleShort: Taking Control of My Data with Local LLM Tools
---

# Taking Control of My Data: Why I Started Building My Own LLM Tools

I keep ending up dependent on other people's tools for things that are fundamentally just data processing: tracking my own tasks, taking notes on my own meetings, writing my own documentation. Every one of those tools wants my data to live in its cloud, in its format, under its subscription. So I started building my own — small, local, LLM-powered tools that do the filtering, summarizing, and organizing myself, on my own compute, over my own files.

That's the actual point of [`llm-playground`](https://github.com/Neo-X/llm-playground). But before any of that automation is trustworthy, I needed to answer a more basic question: which LLMs can I actually run well on the compute I have access to? And "well" is not just "fast." A model that generates 80 tokens/sec but hallucinates task details or garbles a meeting transcript is worse than useless for this — once a tool is managing your data, accuracy and reliability matter more than raw throughput. Speed only matters among the models that are actually good enough to trust.

So the first thing I built wasn't a task tracker or a note-taker — it was a benchmarking toolkit to answer, with data instead of vibes, which local models and hardware setups were even viable candidates. This post is about that foundation. Once I had a model I trusted and compute I understood, I started building the actual data tools on top of it:

- **Task tracking with context** — storing not just "what needs doing" but enough surrounding context that I can pick a task back up cold, days later, without re-deriving where I left off.
- **Meeting transcript processing** — turning raw transcripts into notes and, more importantly, surfacing connections across meetings over time, which has been a real upgrade for research hygiene.
- **Documentation** — for code, and for the seemingly endless bureaucratic paperwork that comes with university life.

I'll write dedicated posts on each of those — how they're built and what they've actually changed about how I work. This post covers the part that made all of them possible: figuring out which local model/hardware combination was good enough to build on.

## The benchmarking foundation

Running LLMs locally (for coding agents, experiments, or these data tools) means constantly asking the same questions: Which model actually fits on this GPU? Is Ollama or llama.cpp faster for this quantization? What batch size gives the best prefill speed on this machine? The answers change with every new laptop, GPU driver, or model release, and "it feels faster" isn't a great basis for a decision — especially when the downstream use is trusting the model with your own tasks and notes.

`llm-playground` is a small collection of Python scripts whose whole job is to turn "which local LLM setup should I use?" into a data problem: run consistent benchmarks, log everything, rank the results, and plot them so the comparison is obvious at a glance. Speed metrics (below) are only half the picture — in practice I pair these benchmarks with manual accuracy checks on the actual task (does it track a task correctly, does it summarize a transcript faithfully) before a model earns a place in my day-to-day tools.

## The core problem: too many knobs, no ground truth

Local inference has a lot of moving parts — backend (Transformers vs Ollama vs llama.cpp), device (CPU, CUDA, ROCm/AMD), quantization, batch size (`-b`/`-ub`), context length, flash attention on/off. Every combination changes both **prefill** (prompt/prefill token speed) and **decode** (generation token speed), and they don't move together. A setup that prefills fast can still generate slowly, which matters a lot for agentic coding workloads where prompts are long and outputs are comparatively short.

The tools below are built around two consistent metrics logged everywhere: `prefill_tps` and `decode_tps`.

## 1. Benchmarking a single backend/model

`benchmark_llm_speed.py` is the base measurement tool. It runs a model (Transformers or Ollama backend) against a fixed prompt, times the prefill and decode phases separately, and appends the result to `logs/benchmark_metrics.csv` / `.jsonl`.

```bash
python benchmark_llm_speed.py \
  --backend ollama \
  --model qwen2.5:3b \
  --runs 5 \
  --warmup 1 \
  --max-new-tokens 128
```

It supports forcing a specific device (`--device cuda|amd|cpu|auto`), auto-pulling Ollama models before the run, and pointing at a non-default Ollama host — useful when the model lives on a remote GPU box.

## 2. Ranking multiple models against each other

Running `benchmark_llm_speed.py` one model at a time doesn't answer "which of these five models should I actually use?" `rank_ollama_models.py` wraps it to benchmark a whole list of models back-to-back and print a ranked table sorted by prefill or decode speed:

```bash
uv run python rank_ollama_models.py \
  --models qwen2.5:3b llama3.2:3b glm-4.7-flash gpt-oss:20b gpt-oss:120b \
  --runs 3 --warmup 1 --max-new-tokens 256 \
  --rank-by decode --device rocm --ollama-pull \
  --csv logs/benchmark_metrics_rocm.csv
```

Rankings land in `logs/ollama_model_rankings.csv`, which becomes the source of truth for "what's actually fast on this machine" instead of relying on memory or vibes from the last time I tried something.

## 3. Sweeping server settings for llama.cpp

For llama.cpp specifically, batch size (`-b`/`-ub`) and flash attention have an outsized effect on prefill speed, and the best value is hardware- and model-dependent. `bench_server_settings.py` automates the sweep: it launches `llama-server` with each batch-size configuration, sends prompts of increasing length, reads prefill tok/s straight out of the server's own timing data, then tears the server down before moving to the next config.

```bash
uv run python bench_server_settings.py \
  -m $MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --container llama-vulkan-radv \
  --port 8001 \
  --flash-attn
```

Output goes to `<model-stem>-server-settings/`: raw `results.jsonl`/`.csv`, `pp_tps.png` and `tg_tps.png` plots (prefill/decode tok/s vs. prompt length, one line per config), and a `README.md` with the best value in each column bolded — so I don't have to re-derive the winning config by eye every time.

For sweeping across llama.cpp *backends* (not just settings) there's also `llama-cpp-bencher.py`, which wraps the upstream `llama-bench` binary and produces the same kind of `results.jsonl` + summary README + plots for a given model.

## 4. Turning logs into comparison plots

Numbers in a CSV are hard to compare at a glance across many runs. `plot_cpu_gpu_comparison.py` takes the per-run or ranked-model CSVs and produces a chart comparing CPU vs. GPU (and optionally AMD) speed:

```bash
python plot_cpu_gpu_comparison.py \
  --cpu-csv logs/benchmark_metrics_cpu.csv \
  --cuda-csv logs/benchmark_metrics_cuda.csv \
  --amd-csv logs/benchmark_metrics_amd.csv \
  --out-png logs/cpu_cuda_amd_speed.png
```

`plot_ollama_rankings.py` and `plot_server_settings_bar.py` do the analogous thing for the ranking and server-sweep outputs, respectively — same idea, different data source.

## 5. Making remote GPU boxes easy to reach

A lot of the actual benchmarking happens on a remote machine with a real GPU, not my laptop. A handful of shell scripts (`setup_llm_distrobox_remote.sh`, `connect-remote.sh`, `connect-remote-llm.sh`, `launch_local_llm_remote.sh`) handle standing up Ollama/llama.cpp on the remote box, opening an SSH tunnel (with Kerberos renewal baked in), and forwarding the right port so a local tool like OpenCode or Claude Code can point at `http://127.0.0.1:11435/v1` as if the model were running locally. A `.bashrc` helper (`ollama-remote`) wraps common commands (`list`, `pull`, `run`) so the tunnel management is invisible day-to-day.

## Why this comes before the data tools

None of these benchmarking scripts are individually complicated — they're mostly `subprocess` calls and CSV rows. The value is in having **one consistent, logged format** for every benchmark I've ever run, across backends, devices, and machines, so when a new model or a new laptop shows up, the question "is this actually good enough to trust with my tasks and notes?" is a quick rerun and a look at a plot, not a rabbit hole of manual testing. `LLMs performance analsys.md` in the repo is the running notes doc where I distill these results into actual hardware/model recommendations (e.g., what fits and runs well on a 128GB unified-memory laptop vs. a discrete 96GB VRAM workstation) — the benchmarking tools are what keep that document honest.

This is the boring-but-necessary layer underneath the tools I actually care about: the ones that track my tasks with enough context to resume them cold, turn meeting transcripts into notes and cross-meeting connections, and handle documentation for both code and university bureaucracy. Those are all just data processing and filtering — the kind of thing I used to hand off to other people's SaaS tools — and now they run locally, on models I've actually vetted, over data that stays mine. Dedicated posts on each of those are coming.

The code is at [github.com/Neo-X/llm-playground](https://github.com/Neo-X/llm-playground) if you want to adapt the same approach to your own local LLM setup.
