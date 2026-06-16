---
layout: post
title:  Generative Error Correction for Pashto ASR
date:   2026-06-16 00:00:00 +0100
categories: 
description: "A simple generative error correcting ensemble model that cuts Pashto ASR error rate by 15 points over the best input model baseline."
---

In a low-resource setting one may sometimes be forced to try to squeeze more performance out of the data and models one has. Model ensembling is one of the first approaches to consider to generically improve performance. For classifiers or regressors, ensembling is straightforward. But for discrete structures such as strings it is not as clear what to do.

In this post I discuss Generative Error Correction (_GEC_), which can be seen as a way to ensemble over models generating strings. This can be seen as analogous to hypothesis rescoring, but it is more powerful. Hypothesis rescoring typically relies on n-best lattice decoding from a single model, which can fail to deliver sufficiently diverse samples. It also limits its output set to be an element of the input set, while a GEC model can in theory weave together the most accurate spans wherever they occur. 


### GEC model

I choose to use ByT5. The advantage is its tokenizer-free approach. There are other reasons to first reach for ByT5, including invalid semantic "jumps" that token-based pretrained models can make when used in error correction settings, that byte-based models avoid[^3]. The disadvantage of the model is the increased sequence lengths, especially for non-Latin script, contributing quadratically to input processing and decoding latency[^1].

So, as a simple experiment I take a Pashto[^2] ASR dataset and measure the performance improvement relative to a selection of multilingual ASR models. 

### ByT5 Prompt Template (Multi-Source Ensemble)

As en example: the input to our GEC model consists of the transcriptions coming from five input models. The expected output for the model to output is the corrected transcription. It is trained using the Huggingface seq2seq trainer.

```
dolphin: نن یو سوه ښتی لهخو به پاسسېدم
mms: نن یو سه وختی له خو به پاسی دم
omniasr: نن یو سه وختی لہ خو بہ پاسېدم
seamless: نه نیوسه وښتی له خود به پاسه یېدم
whisper-v3: نن یو سو اختیل خوب پاسدم
correct: نن یو څه وختي له خوبه پاڅیدم.
```

This includes the same four models as this paper[^4], as well as _Dolphin_, which is another multilingual ASR model capable of transcribing Pashto. 

| Model | HuggingFace | Repository |
|-------|-------------|------------|
| Dolphin-small | [DataoceanAI/dolphin-small](https://huggingface.co/DataoceanAI/dolphin-small) | [DataoceanAI/Dolphin](https://github.com/DataoceanAI/Dolphin) |
| MMS-1B | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) |
| OmniASR-CTC (300M) | [Steveeeeeeen/omniASR-CTC-300M](https://huggingface.co/Steveeeeeeen/omniASR-CTC-300M) | [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication) |
| SeamlessM4T-v2 | [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication) |
| Whisper-large-v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [openai/whisper](https://github.com/openai/whisper) |


I use this dataset: [Common Voice Scripted Speech 25.0 (Pashto)](https://mozilladatacollective.com/datasets/cmndf6mgs001lnz07bf9t3skp)  
~5,314 hours of read Pashto speech from 8,239 speakers.  
Test split: 15,462 validated clips.  Dev split: 15,462 validated clips.

The GEC model is trained on the dev split and evaluated on the test split.


## Pashto ASR Evaluation - CV25 Test Split (N=15,462)

### Per-Model WER / CER

| Model | WER | CER |
|-------|-----|-----|
| Dolphin-small | 52.2% | 24.2% |
| MMS-1B | 69.3% | 29.9% |
| OmniASR-CTC (300M) | 49.3% | 19.7% |
| SeamlessM4T-v2 | 46.8% | 23.9% |
| Whisper-large-v3 | 92.6% | 41.1% |
| **ByT5-base (text-only fine-tuned)** | **34.4%** | **14.7%** |

Our GEC model performs better than any model individually.

## Efficiency

Combining models and training an error correction model is an increase in operational complexity. A more thorough analysis might try to determine if most of the benefit can come from selecting one or two models rather than five. Perhaps one could find a smaller set of models that perform well individually while being still being uncorrelated in their errors.

### CER Correlation Matrix

| | Dolphin | MMS-1B | OmniASR-CTC | SeamlessM4T-v2 | Whisper-v3 |
|---|---------|--------|-------------|----------------|------------|
| Dolphin-small | 1.00 | 0.68 | 0.77 | 0.41 | 0.68 |
| MMS-1B | 0.68 | 1.00 | 0.72 | 0.39 | 0.64 |
| OmniASR-CTC | 0.77 | 0.72 | 1.00 | 0.42 | 0.74 |
| SeamlessM4T-v2 | 0.41 | 0.39 | 0.42 | 1.00 | 0.39 |
| Whisper-large-v3 | 0.68 | 0.64 | 0.74 | 0.39 | 1.00 |

As an example, _OmniASR-CTC_ and  _SeamlessM4T-v2_ are the best performing individual models while having a relatively low error correlation of 0.42.

### Single model GEC approach

One could even use a single model as input to an error correction model. This could happen in two ways - firstly, the error correction model could simply learn to correct frequent error types from single-transcript input. Second, one could sample from the model. _OmniASR-CTC_ has the distinction of being the fastest model for inference amongst those tested, and having the best error rate. Unfortunately, there is no principled way to sample from a CTC model. Yet one can approximate sample generation through noise injection.

#### Oracle N-Best Sampling

To give an idea of what single-model sampling might achieve, I do the following: on a small set of transcripts, I generate 10 samples each, either using temperature-based stochastic decoding, or via noise injection (5% TimeMask gave diverse samples without too much raw error increase). Then, I calculate the minimum CER value with respect to the expected transcript. So, this oracle selection technique might give a rough idea of how much our GEC model might expect to improve error rates if it had been given these inputs.


| Model | 1-sample WER | 10-sample min WER | WER Δ | 1-sample CER | 10-sample min CER | CER Δ |  Sampling |
|-------|-------------|--------------|-------|-------------|--------------|-------|----------|
| SeamlessM4T-v2 | 30.9% | 24.0% | −22.5% | 14.5% | 9.8% | −32.8% | stochastic (temp=0.6) |
| OmniASR-CTC (300M) | 37.9% | 33.4% | −11.9% | 13.4% | 11.8% | −12.1% | noise injection (TimeMask 5%) |

The oracle n-best estimate of e.g. ~33% CER reduction for _SeamlessM4T_ is both optimistic and pessimistic. Optimistic, because we take the minimum CER over the 10 samples, but pessimistic because our model could correct errors beyond simply selecting the min-CER hypothesis.

The expected sampling benefit for OmniASR is lower, probably because the noise injection sampling generally leads to higher error rate even if a small number of individual samples may have a better error.


### GEC on finetuned models

In the _Benchmarking Multilingual Speech Models on Pashto_ paper, several finetuned models are discussed, which individually achieve error rates comparable to the GEC model. Further analysis might consider if these also benefit from a GEC approach, when added as input models. There would likely be diminishing returns but the exact extent would be interesting to know.

## Discussion

So, why would one do this rather than just finetuning? For one thing, while I don't consider it here, there may be contextual information that could be included in the prompt which may provide clues to the correct transcription that could be difficult to incorporate into a traditional ASR model.

Second, training a GEC model might be a viable part of a pseudo-label generation strategy where unlabeled data is plentiful, and retention of raw audio or audio features is problematic.

[^1]: To overcome the latency difficulties, ByT5 would be an excellent target for adaptation into a diffusion language model, which apparently is possible for such encoder-based models. For an example of diffusion LLMs' relative latency in the error correction setting, take a look at [this experiment](https://huggingface.co/buckets/davanstrien/diffusiongemma-ocr-bench) for OCR.

[^2]: nothing in this analysis is specific to Pashto except it being an example low-resource language. 

[^3]: ["Byte-Level Grammatical Error Correction Using Synthetic and Curated Corpora"](https://aclanthology.org/2023.acl-long.402.pdf), Li et al., ACL 2023.

[^4]: ["Benchmarking Multilingual Speech Models on Pashto"](https://arxiv.org/abs/2604.04598).