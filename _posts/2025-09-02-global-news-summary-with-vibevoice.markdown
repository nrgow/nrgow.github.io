---
layout: post
title:  "(Mostly) Self-Hosted Global News Summary with VibeVoice"
date:   2025-09-02 09:50:00 +0200
categories: 
---

Is it possible to listen to global news while it is happening? That might sounds like a strange question - there are plenty of news channels or radio stations one could listen to. Let's just say that there may be some value in having access to news data and the means of aggregating and presenting it.

Of course, to maximize the efficiency of ingesting said news summaries, we need to be able to generate nice spoken audio. [VibeVoice](https://github.com/microsoft/VibeVoice), the new long-form TTS generator from Microsoft looks like a promising candidate to test.

[GDelt](https://www.gdeltproject.org/) amazingly still publishes all its news data. In theory, every minute, they provide a list of news articles they have collected, including the headline and the url. They do lots of other really interesting analysis, but for this project I'll just look at the headlines. 

Let's self-host as much of this as possible. The constraints are thus: one 4090 GPU with 24GB VRAM and 128GB of regular RAM.

There is still some noise in the headline data, including some incorrect language detection. So we'll need a second opinion (actually also a third opinion as well to enable tie breaking). Other cases of noise include "boilerplate", that is including metadata about the source in the headline itself. There is also irrelevant non-news data.

To maximize diversity of information sources, we also need to be able to handle non-English news. Of course a sufficiently strong LLM could just generate a summary from the original language headlines. However, to filter noisy data and thus to be economical with LLM context, as well as enabling other sorts of enrichment, I think it's easier if we have a version of the headlines translated to English.

Let's self-host the translation model. The quantized Bytedance Seed-X model, [ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8](https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8), fits on the GPU, and in my very short eyeballing of its output, I would say it works pretty well. In any case the English text is mostly plausible. Tencent also released (yesterday) a similarly sized MT model, [Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B) and a [translation ensemble model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8), which sounds very interesting. I may have to test these as well.

For filtering, after looking at some pretrained news classifier models which worked ok but not great, I didn't feel like coming up with a clever scheme for combining or handling their predictions. So instead, let's do zero-shot classification, and simply track the "probability" of the article being about "international politics". The zero-shot formulation also potentially leads to some cool custom topic-based news summaries.


![news pipeline]({{ "/assets/images/news-pipeline.png" | relative_url }})


To generate the transcript I had a moment of weakness and just decided to use a cloud LLM via OpenRouter. I use __gemini-2.5-flash-lite__, for no particular reason. 

Here is an example transcript:

> Speaker 1: Welcome to Live World News. It is Monday, September 1st, 2025, 4:57 PM.  
> Speaker 2: Today's top stories. The United Nations reports that 850,000 Syrian refugees have returned home since the fall of Assad.  
> Speaker 1: In the United States, a judge intervened to halt the deportation of dozens of Guatemalan minors.  
> Speaker 2: Venezuela has fired upon a ballot boat belonging to Guyana ahead of a national vote.  
> Speaker 1: Alexei Navalny's body has been released to his mother.  
> Speaker 2: Uzbekistan is emphasizing the strategic importance of transport corridors within the Shanghai Cooperation Organization.  
> Speaker 1: Reports indicate a Houthi security apparatus has abducted a prominent foreign UN official in Yemen.  
> Speaker 2: In Thailand, the People's Party could play a decisive role in selecting the next Prime Minister.  
> Speaker 1: Donald Trump is warning a court against ending his tariffs plan.  
> Speaker 2: Yemenis are mourning their Houthi prime minister following a rebel attack on a ship in the Red Sea.  
> Speaker 1: The Shanghai Cooperation Organization summit has concluded, with leaders like Xi Jinping and Vladimir Putin asserting a new vision for global governance, often in contrast to Western policies.  
> Speaker 2: Tensions remain high in the Middle East as a flotilla attempting to reach Gaza was forced back to port due to weather conditions.  
> Speaker 1: Media outlets are protesting the deaths of journalists in Gaza and demanding access.  
> Speaker 2: Concerns are also being raised about potential Russian interference with the navigation systems of an EU leader's plane over Bulgaria.

I won't claim that this is particularly riveting or even insightful beyond a simple readout of some headlines. But it is truly globally based, and quite fresh too. I have seen some better examples, and some prompt refinement will doubtless help. I'd like the LLM to be able to generate some analysis and discussion. One way could of course be to provide the full text of the articles. Another way would be to provide the LLM with knowledge of recent world events, either in the prompt or via some other temporal knowledge update mechanism. Asking for the LLM to generate its own analysis without some kind of grounding risks hallucinating events beyond its knowledge cutoff date.

Anyway, VibeVoice performs IMO admirably. Here's what it generates:

{% include embed-audio.html src="/assets/audio/vibevoice-news-audio.wav" %}

Not too bad for a start.