---
layout: post
title:  "Spoken Language Entity Linking: Human Data and Voxtral"
date:   2025-07-16 16:44:00 +0200
categories: 
---

After recording 50 tests sentences for German street names, we can now get a rough evaluation of the system on some approximation of "real data". I should mention that I can speak German - this might matter.

In previous posts I used the sparse character ngram baseline. It struck me that I presented this as a reasonable baseline with motivation but without much evidence. Let's try also try a dense baseline, with the very generic choice of ``sentence-transformers/all-MiniLM-L6-v2``, the model which failed to provide hard negatives. We'll give it another chance, this time as a retrieval model, out of curiosity.

Finally I'll also take a look at ``mistralai/Voxtral-Mini-3B-2507``, a freshly-baked multimodal model, released just yesterday, to for a simple comparison against the other multimodal LLM previously mentioned, ``google/gemma-3n-E2B-it``.


## The evaluation


| retrieval model   |gemma simpleprompt | gemma hintprompt |   parakeet |   voxtral-mini |   whisper-turbo |
|-------------------|------------------|------------------|------------|----------------|-----------------|
| dense-baseline    |             0.06 |             0.16 |       0.16 |           0.28 |            0.42 |
| **dense-v1**      |             0.34 |             0.5  |       0.64 |           0.56 |            **0.74** |
| sparse-baseline   |             0.16 |             0.32 |       0.4  |           0.42 |            0.58 |

*Recall@1*


| retrieval model   |gemma simpleprompt| gemma hintprompt |   parakeet |   voxtral-mini |   whisper-turbo |
|-------------------|------------------|------------------|------------|----------------|-----------------|
| dense-baseline    |             0.08 |             0.26 |       0.24 |           0.38 |            0.52 |
| **dense-v1**      |             0.54 |             0.78 |       0.84 |           0.74 |            **0.92** |
| sparse-baseline   |             0.26 |             0.56 |       0.56 |           0.58 |            0.78 |

*Recall@10*


Here's the evaluation. A few things stand out. While in synthetic data evaluation, parakeet and whisper-turbo performed roughly the same, with human data whisper-turbo performance considerably better, over all retrieval models. I suspect this arises from whisper-turbo's ability to incorporate paralinguistic features, such as the speaker accent, into its decoding process. Already the top-1 performance is looking pretty good.

The dense-baseline performs rather poorly, giving some evidence to the hunch that character ngrams would be a strong baseline. 

Finally, regarding voxtral-mini: it performs pretty well! Better than Gemma-3n-2b with the base prompt, and comparable to the results where Gemma-3n-2b is given the enhanced task prompt including contextual hint. Comparison with parakeet or whisper turbo would be a bit unfair - for either gemma or voxtral - the phonetic embedding model was only trained on synthetic data coming from parakeet and whisper-turbo, so the embeddings are likely better attuned to the types of transcription errors that they make. 

It's a promising model which comes with a very nice vllm integration. Further posts may dive a little deeper into its capabilities. 