---
layout: post
title:  "Spoken Language Entity Linking: Embedding Model"
date:   2025-07-11 11:28:24 +0200
categories: 
---

With out synthetic data prepared we can now train an embedding model and attempt an initial evaluation of the whole system. The embedding model will have a byt5 backbone. Why byt5? Shouldn't the base model by a hyperparameter?  I will state without proof that this task requires a tokenizer-free approach. Standard LLMs with learned subword tokenizers are notoriously bad at understanding the internal structure of their subwords, and the phonetic similarity task requires just such understanding. There are some other tokenizer-free models, such as [canine](https://huggingface.co/google/canine-c), which could be interesting as well. But I'll start with [byt5-small](https://huggingface.co/google/byt5-small), with mean pooling and an embedding dimension of 256. The data is a combination of transcriptions from parakeet and whisper-turbo.

## Baseline model: character ngrams

As mentioned in the introduction post, when going down the road of neural networks, GPUs and training models, we need to first pause and think if all this is really necessary. Perhaps there is a simpler way.

We can try (bag-of) character ngrams.


    {'b': 1, 'a': 1, 'c': 1, 'h': 1, 's': 1, 't': 1, 'e': 3, 'l': 1, 'z': 1, 'n': 1, 'w': 1, 'g': 1, '^b': 1, 'ba': 1, 'ac': 1, 'ch': 1, 'hs': 1, 'st': 1, 'te': 1, 'el': 1, 'lz': 1, 'ze': 1, 'en': 1, 'nw': 1, 'we': 1, 'eg': 1, 'g$': 1, '^^b': 1, '^ba': 1, 'bac': 1, 'ach': 1, 'chs': 1, 'hst': 1, 'ste': 1, 'tel': 1, 'elz': 1, 'lze': 1, 'zen': 1, 'enw': 1, 'nwe': 1, 'weg': 1, 'eg$': 1, 'g$$': 1}

*Bag of character ngrams for "Bachstelzenweg"*

We need to maintain a mapping from ngrams to the index of that vector in the database. Elasticsearch can (implicitly) handle that for us. It's only a minor annoyance, because we can use the hashing trick to simply hash the ngrams and then interpreting the hash as a uint32 we have the indices. It turns out that this is exactly what Qdrant does in their own *fast-embed* library. Hash collisions may occur but they will add little to the overall error rate.

    {1590227222: 1, 3068098019: 1, 3771431720: 1, 4117268401: 1, 2087417677: 1, 1628501593: 1, 2265248077: 3, 2295987441: 1, 2304499609: 1, 1333902333: 1, 3405261443: 1, 3716256635: 1, 267413913: 1, 62056036: 1, 2058349989: 1, 3979491815: 1, 1527449582: 1, 3808550603: 1, 1415237013: 1, 3465992959: 1, 1980614861: 1, 915182080: 1, 3224134014: 1, 3601840990: 1, 1887120473: 1, 1834228204: 1, 1813522373: 1, 123610086: 1, 3593534650: 1, 3740847369: 1, 1353835269: 1, 3920023514: 1, 2705968765: 1, 1310277804: 1, 4271617947: 1, 3836328121: 1, 486360380: 1, 4056173117: 1, 4160623159: 1, 948873334: 1, 833345316: 1, 2802396029: 1, 3026077236: 1}

*Hashing the ngrams and ignoring collisions allows easy insertion to Qdrant*

## Baseline model evaluation

Let's take recall@10 as the evaluation metric - is the right answer in the top-10 results. The task is relatively hard so I don't expect top-1 accuracy to be very good. But we can look at that too. Recall@10 also gives us an upper bound on the accuracy of a reranking approach that reranked the top 10 results.

Recall@10 is around 40-42%. Not great, but maybe better than expected.

One of issues is that the nearest neighbors are often pretty implausible (see table at the end). We cannot expect too much as a bag-of-ngrams loses all ordering information.


## Embedding model 

We only have positive examples of known word-transcription pairs, so we'll need to do negative mining to get negative examples. For want of a better option right now, let's generate them using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.

### First attempt

Everything apparently goes fine during training, saturating at 99.6% accuracy! 

![training-eval-set-accuracy]({{ "/assets/images/eval-set-accuracy.png" | relative_url }})

*Clearly, the model is ready to deploy without further inspection. Or is it?*

So now we can embed our 500k street names and test the accuracy of the resulting vectors. Recall@10 improves, but only by 10 percentage points. Was it all worth it?


### The solution: hard(er) negative mining

At this point, one may ask what went wrong, or abandon the whole enterprise entirely. Maybe something with training, like overfitting, or maybe the synthetic data is just too low quality. After much pondering, the most likely suspect is indeed the training data, but rather, how the negative samples were generated. We used `all-MiniLM-L6-v2` to generate negative examples, but as per the foregoing, it has no notion of phonetics, so the negative examples are unlikely to be very useful.

We need _hard negatives_. Maybe the baseline ngram model could generate them? This could work, but actually I decide to just use the just-trained embedding model itself to generate hard negatives. This v0 embedding model doesn't perform great but it will generate better negatives than MiniLM.

After retraining, the evaluation looks much better. We get recall@10 of 78%. If we are really interested with a single result, recall@1 is 51%, and a reranker could get an upper-bound of 86% if it had access to the top-40 examples. That's more like the kind of delta relative to the baseline I was hoping to achieve. And looking at the nearest neighbors for some examples (table at end of post), the results seem a little more phonetically plausible.


| at_k         | recall@1 |               | recall@10|               | recall@40|               |
| asr          | parakeet | whisper-turbo | parakeet | whisper-turbo | parakeet | whisper-turbo |
|--------------|----------|---------------|----------|---------------|----------|---------------|
| sparse       | 0.21     | 0.22          | 0.40     | 0.42          | 0.53     | 0.53          |
| dense-v0     | 0.29     | 0.31          | 0.51     | 0.52          | 0.63     | 0.64          |
| dense-v1     | 0.51     | 0.52          | 0.78     | 0.78          | 0.86     | 0.86          |

*The scores for different models and metrics on the validation set*

## Outlook
We now perform significantly better than the baseline, which leaves us at a crossroads. Where to go next to improve the system?


### Reranking (two-stage search)
We can just as easily train a reranker base on a biencoder. Reranking from 40 candidates could give us _up to_ 35 percentage point improvement in top-1 accuracy.


### Ensembling over sparse/dense vector models (hybrid search)
The dense model does not strictly dominate the sparse model: there are cases where the correct result is in the top-10 sparse results but not in the dense, and vice-versa. So there's possibly something to be gained from a combination. As a rough upper-bound, for e.g. the parakeet ASR model, we could check if the correct result in either the sparse top-10 _or_ the dense top-10. That would give us 80%.


### End-to-end ensembling over ASR models
The idea is this: if two ASR models make somewhat uncorrelated errors (or! if one model could be induced to provide a _diverse_ nbest list), then, for a given user query, we could run both models, and generate embeddings for both transcripts. The two vectors could then be averaged. These averaged vector will have lower variance than those coming from a single model. We can quickly compute something like an upper-bound for how much better our recall@10 would be, by asking per-entity, is the right answer in the top-10 result for either asr model. This would give 86%. So there is potentially more to gain from ensembling over ASR transcriptions, rather than over vector search methods, and could improve results roughly as much as a reranker could. These results are not so precise because we are would now be considering up to 20 items rather than 10, and secondly, the results from the averaged embedding are not always coming from the union of the results of the two input vectors.


### Retrieval-augmented audio-text hypothesis rescoring
Now that we have a passable retrieval model, we could try using a multimodal llm to reanalyse the audio given 10 likely transcription candidates. The idea being, either zero shot, or via finetuning, to use the audio LLM as a reranker that not only has access to the candidate texts, but also the audio. A multimodal LLM gives us this potential and more.


## Speed
I can load 3k sparse vectors per second into Qdrant. Queries reach 300-200 per second. I can load 1.3k dense vectors per second, and that includes the time to embed on my local GPU. I made some use of the [batch insert performance guide](https://qdrant.tech/articles/indexing-optimization/), but things could be optimized further. Querying reaches 50 per second after the HNSW index hash finished construction.


## What's next?
In the next post I'll reveal how this system performs on the human control data.



## Example nearest neighbors for each retreival system

| dense-v1              | sparse           |
|-----------------------|------------------|
| Salzhübelstraße       | Zubergasse       |
| Zöblitzer Schulstraße | Zaubergasse      |
| Zollstrasse           | Zalm             |
| Zollstadlstraße       | Deubelsgasse     |
| Zabelstraße           | Geubelsgasse     |
| Zellstraße            | Zaungasse        |
| Zerzabelshofstraße    | Zahlgasse        |
| Zelzater Straße       | Zaltbommelstraat |
| Zobelstraße           | Zalmstael        |
| Zitzelsbergerstraße   | Zabelstraße      |

_Street: Salzhübelstraße, ASR transcription (parakeet): Zaltzubelstasse_


| dense-v1          | sparse              |
|-------------------|---------------------|
| Im Försterkamp    | Im Kamp             |
| Im Pferdekamp     | Im Karkamp          |
| Im Forstkamp      | Schmitz Kamp        |
| Im Stadtkamp      | In de Kamp          |
| Auf dem Pützacker | In de Kamp          |
| Auf dem Osterkamp | Ihsrader Kamp       |
| Auf dem Strickart | Im Hof Kamp         |
| Am Försterkamp    | Wilder Kamp         |
| Am Forstkamp      | Im Nesselrader Kamp |
| Am Pferdekamp     | Im Huckinger Kamp   |

_Street: Im Försterkamp, ASR transcription (parakeet): Imfutz der Kamp_


| dense-v1       | sparse         |
|----------------|----------------|
| Stallgatan     | Stadl          |
| Stellagatan    | Stadler Garten |
| Stålgatan      | Sologatan      |
| Stadler Garten | Salogatan      |
| Stallergarten  | Silogatan      |
| Stollergarten  | Stadler        |
| Stoltsgatan    | Logatan        |
| Stilla gatan   | Stadlhof       |
| Stilla Gatan   | Stadlwiesen    |
| Sattlergang    | Stadlweg       |

_Street: Stadler Garten, ASR transcription (parakeet): Stadlogaten_