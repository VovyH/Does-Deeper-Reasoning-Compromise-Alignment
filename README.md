**Does Deeper Reasoning Compromise Safety Alignment? Revealing and Mitigating Alignment Collapse in Large Reasoning Models**

We study whether extended reasoning in Large Reasoning Models (LRMs) makes models more vulnerable to external perturbations. Our experiments show that deeper reasoning can improve clean-task performance while increasing sensitivity to irrelevant noise, scenario nesting, and adversarial jailbreak prompts.

## Overview

Large Reasoning Models are often expected to become safer and more reliable as their reasoning depth increases. However, our study reveals a different phenomenon: extended reasoning may amplify the effect of perturbations and weaken the model's adherence to the original input or safety constraints.

We refer to this phenomenon as **Alignment Collapse under Deep Reasoning**.

The repository includes:

- controlled perturbation experiments on reasoning tasks;
- safety evaluation on jailbreak benchmarks;
- Reasoning Trap (RT), a reasoning-triggered framework for exposing safety risks;
- Reasoning Residual Alignment (RRA), a lightweight mitigation strategy;
- experimental results and analysis data.

## Repository Structure

```text
hallucination_jailbeak/
├── result/
├── data/
└── hallucation/
    └── Baseline/
```

### Directory Description

- `hallucination_jailbeak/result`: stores the experimental results.
- `hallucination_jailbeak/data`: contains the data used for analysis.
- `hallucination_jailbeak/hallucation`: includes the implementation details of all evaluated testing methods.
- `hallucination_jailbeak/hallucation/Baseline`: contains the baseline jailbreak attack methods.

> Note: the directory names follow the current repository structure.

## Main Components

### Alignment Collapse Evaluation

We evaluate how reasoning depth affects robustness under external perturbations. Given a clean input and its perturbed version, we compare model performance across different reasoning budgets.

We use **Alignment Loss Rate (ALR)** to quantify the relative degradation caused by perturbations:

```text
ALR = (Clean Accuracy - Perturbed Accuracy) / Clean Accuracy
```

A higher ALR indicates stronger vulnerability to perturbations.

### Reasoning Trap

**Reasoning Trap (RT)** is a reasoning-triggered framework for revealing safety risks in standard LLMs. RT combines existing jailbreak prompts with a reasoning trigger to test whether extended reasoning amplifies adversarial perturbations and weakens refusal behavior.

RT is designed as a diagnostic framework rather than a standalone jailbreak method.

### Reasoning Residual Alignment

**Reasoning Residual Alignment (RRA)** is a lightweight mitigation strategy. It re-emphasizes the original input after the reasoning process, reducing the dilution of safety-relevant constraints during long reasoning.

## Experiments

The repository contains experiments for:

- reasoning-task perturbation robustness;
- jailbreak robustness under different attack methods;
- RT-enhanced jailbreak evaluation;
- attention-based analysis of reasoning-induced vulnerability;
- RRA mitigation evaluation.

The evaluated attack baselines are stored in:

```text
hallucination_jailbeak/hallucation/Baseline
```

The experimental outputs are stored in:

```text
hallucination_jailbeak/result
```

The analysis data are stored in:

```text
hallucination_jailbeak/data
```

## Models and Benchmarks

Our experiments involve multiple reasoning and standard LLMs, including Qwen, DeepSeek, Llama, GLM, and Claude-series models.

The benchmarks include:

- reasoning benchmarks for task robustness evaluation;
- AdvBench for safety and jailbreak evaluation.

## Usage

Please first prepare the required model APIs or local model checkpoints according to your experimental setting.

The core experimental methods are organized under:

```text
hallucination_jailbeak/hallucation
```

Baseline attack methods can be found in:

```text
hallucination_jailbeak/hallucation/Baseline
```

After running the experiments, results should be saved to:

```text
hallucination_jailbeak/result
```

## Results

The main results show that:

1. deeper reasoning improves clean-input reasoning performance;
2. deeper reasoning increases vulnerability to external perturbations;
3. reasoning-triggered prompts can amplify the effectiveness of existing jailbreak attacks;
4. re-emphasizing the original input can partially mitigate reasoning-induced safety degradation.
