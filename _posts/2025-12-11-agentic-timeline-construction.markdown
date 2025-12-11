---
layout: post
title:  Agentic Timeline Construction
date:   2025-12-11 12:40:00 +0100
categories: 
---

In a previous blogposts I looked at a simple way to induce a probability distribution for future binary events as _timeline continuation_. Let's temporarily suspend any doubt as to whether this might ultimately be really viable. I can also already see weaknesses about the strategy. First, I expect the predictions to be highly sensitive to the content of the initial timeline. Secondly, for unlikely events, most simulated timelines are unlikely to any information about the event under question, without directly including it as a instruction for generation. So why not just ask the model to simulate scenarios for that rare event? Certainly an option, but my feeling is to prefer to separate what a model _believes (superficially?)_ from the beliefs _implied_ by the model's understanding of historical/etc dynamics.

With that said, if we want to investigate further, we will need a way to generate timelines relating to topics, containing all information up to the present. 

Timeline construction is more or less a subgenre of deep-research, additionally with chronological structure.

The agent is pretty simple:

1. ReAct agent constructs initial sketch timeline.
2. Timeline parsing LLM call extracts events from the initial timeline.
3. For each event extracted from the initial timeline, another ReAct agent "deepens" information relating to that event.
4. Timeline-merging LLM call constructs the finished timeline aggregating all the events that have been researched in depth.

I start with two tools for the ReAct agent: first, a wikipedia search tool and wikipedia page tool. Second, I try the GDelt _document search_ api. It turns out this api has a few limitations: basic terms shorter than 4 characters cannot be searched. Search terms cannot be refined by AND combination; only OR is supported. And of course the full text content is not available, only headlines, limiting the depth that can be added by this API.

The next step would be to add Exa MCP or another web search tool. In this scenario I am trying to ground the future event likelihood in prediction markets, which can move quickly. So efficiency would be important. I tend to think that future timelines could be more efficiently generated simply as a RAG direct from a suitable prepopulated event DB. Then, the difficulty would be to construct the DB such in a way that with a relatively low-depth deep search, all relevant events could be retrieved.

I'll hold off on demonstrating an example of this brute-force timeline generation (code [here](https://github.com/nrgow/timeline-prediction/blob/main/src/generate_timeline_to_now.py)) because the next blogpost awaits.