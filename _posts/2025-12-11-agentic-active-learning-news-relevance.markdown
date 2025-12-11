---
layout: post
title:  Agentic Active Learning for News Relevance
date:   2025-12-11 13:40:00 +0100
categories: 
---

As I mentioned in the previous post, constructing a timeline can mostly be brute forced by an agent which is able to crawl relevant content from wikipedia or elsewhere. Of particular interest for the "future prediction"/prediction market grounding scenario is the most recent data that might not at all be referred to by existing documents. Access to a real commercial search index might have sufficient data, but let's see what we can build.

Looking at the gdelt data over a 24 hour period, taking English language non-duplicated data, it only amounts to about 100k headlines. Sufficiently small that we might try some more fancy approach than just embedding-based cosine similarity.

Active learning basically refers to a setup with a human-in-the-loop. For a series of iterations, the human will annotate a small sample of items, sampled in some way reflecting the uncertainty of the classifier in order to maximize the utility of the annotations with respect to the current set of annotations, then the classifier is retrained and new predictions are made.

Naturally now the human annotator in the loop can be replaced by an LLM.

Why active learning instead of any other relevance approach? Here, the number of items is pretty small, so for each query we can train a separate NN on top of the embeddings. So it's nice to be able have a more flexible class of decision boundaries than what cosine similarity can offer, without having to regenerate the embeddings. Another nice thing is that we are adapting the classifier to the actual dataset, so it might be more robust to data distribution shift. One negative is the sensitivity to the initial seed data. A final nice thing is that at the end you have a classifier which could be deployed for further inference on new news.

The choice of sampling will very important for the success of this enterprise. For queries which naturally match many documents, uniform-ish sampling will return all negative. I had Codex cook up some sampling strategies. I need some evaluation harness to work out which one is best.

Inference is done by exporting the model to onnx and inferring on an mmap'd binary file containing the embeddings. I didn't find any benefit at this scale to further exporting the model to tensorrt-llm (which takes a couple of seconds). As a humorous aside, a codex-generated GPU diagnostic script revealed that a botched M2 disk insertion had knocked my GPU down to PCIE v1. I'd like to scale this maybe up to 10 million embeddings but some tricks will be required to keep things in the reasonable timeframe for an agent tool call. The time cost is currently dominated by the LLM annotation calls rather than training or inference; these can of course be parallelized. In any case, the success of LLMs has demonstrated that people are willing to wait relatively long to get information, if the information is good.


![search results for news relating to German economic decline]({{ "/assets/images/search_results.png" | relative_url }})

*Search results for news relating to German economic decline*

Looking at the current results - hits for the phrase _"News relating to German economic decline"_ - precision could be improved, I think by focusing the active learning annotation sampling at higher-scoring results. And recall I think is probably missing related news where the keyterm Germany is not mentioned. This might be due to weak keyphrase expansion at the beginning, or maybe the LLM annotator needs to be prompted a bit better to include indirectly relevant hits. 

The final piece of the puzzle, assuming this can be made to work well enough would be actually going from the relevant articles obtained hereby to a succinct timeline update. The code is [here](https://github.com/nrgow/agentic-active-learning-news-relevance).