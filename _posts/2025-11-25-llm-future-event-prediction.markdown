---
layout: post
title:  Generation strategies for future event probability estimation
date:   2025-11-25 15:44:55 +0100
categories: 
---

Can LLMs predict future events? Of course they can. On the other hand, it's not totally obvious what is the best method actually get those prediction out from the models. In this blogpost, a prediction means deriving a probability of an event taking place.

Let's take as an example the following prediction market: [Russia x Ukraine ceasefire by end of 2026?](https://polymarket.com/event/russia-x-ukraine-ceasefire-before-2027?tid=1764068376362). We could ask a model directly for a probability estimate. Or, we could ask the question multiple times then count the distribution of answers.

The approach I try is this:

1. generate a freeform timeline "_into the foreseeable future_" on the topic ("Russia/Ukraine conflict").
2. in a second LLM call, ask whether the generated timeline implies that the question resolves to true.

Additionally, because LLMs have knowledge cutoffs, when generating the timeline, I boost the context with relevant Wikipedia articles containing info the models may not have seen before, namely these: 

- [Timeline of the Russo-Ukrainian war (1 September 2025 – present)](https://en.wikipedia.org/wiki/Timeline_of_the_Russo-Ukrainian_war_(1_September_2025_%E2%80%93_present))
- [Peace negotiations in the Russo-Ukrainian war (2022–present)](https://en.wikipedia.org/wiki/Peace_negotiations_in_the_Russo-Ukrainian_war_(2022%E2%80%93present))

Choosing by walking down the [openrouter model list](https://openrouter.ai/models?fmt=cards&input_modalities=text&output_modalities=text), I select these models:

- anthropic/claude-opus-4.5
- google/gemini-3-pro-preview
- openai/gpt-5.1
- openrouter/bert-nebulon-alpha
- x-ai/grok-4.1-fast:free

Then, I generate for each model a timeline at these temperatures: __[0.1, 0.3, 0.5, 0.7, 0.9]__.

With 5 models and 5 temperatures, we get 25 predictions. 10 return false, 15 return true. **This indicates a 60% probability of ceasefire by the end of 2026**. What is the "ground truth"? In the past 24 hours the implied probability rose from 50% to 58%.

![The market implied probability]({{ "/assets/images/polymarket_ukraine_russia.png" | relative_url }})


Actually, I was planning to develop this system even further, but I find its prediction to be already surprisingly close to the Polymarket estimate, so maybe better to stop here than to continue and find some evidence invalidating the technique.

This particular prediction market has the benefit of having well-structured Wikipedia timeline articles on relevant events. This won't be true of all markets. So, generating the contextual timelines from news to the present might be an interesting thing to do.

Code is [here](https://github.com/nrgow/timeline-prediction/). It's not strictly necessary to use dspy here as I have no immediate intention of trying to optimize the prompts for this.