---
layout: post
title:  GRPO and audio embeddings for TTS style control
date:   2025-11-07 17:00:00 +0200
categories: 
---

I had been casually looking for an introductory project for reinforcement learning to get some basic familiarity with the subject, and came across [this blog post](https://huggingface.co/blog/Steveeeeeeen/llasa-grpo).
It describes a relatively straightforward RL optimization of a TTS model, without requiring only textual data, based on a combination of two rewards: first, the word error rate of the synthesized signal after applying speech to text, and secondly the negative log likelihood of the text given the synthesized audio (with the likelihood computed by a whisper model). The interesting point is the specific TTS model architecture, Llasa, is a decoder-only model based on discrete tokens - this allows using existing RL frameworks and algorithms without any modification.

So how might we extend this? My first thought was to try to synthesize "L2" speech, i.e. English language speech with, for example, a German accent. The idea being to apply spoken language identification to the generated speech signal, then to add two new rewards: (1) the probability of English, and (2) the probability of German. To keep things short, I think that might be too ambitious, especially as the Llasa model is only trained on English and Chinese, I think it's not so likely a model could "invent" the German accent just based on RL rewards.

So a more modest task is simply speaker style. The idea here is thus: try to generate speech in a _whispering voice_. For the score I do the following: I use the CLAP model, which is a joint audio/text embedding model, and generate an embedding from the text: _"a man whispering"_. The reward is then simply the cosine similarity between that text embedding, and the embedding of the synthesized audio.

![some training runs, showing the initial configurations which diverged, and later runs with more stable reward growth]({{ "/assets/images/rl_for_tts.png" | relative_url }})

*Some training runs, showing the initial configurations which diverged, and later runs with more stable positive reward trajectory*


A few other modifications were necessary. First, I wanted to run this at home on a "small" gpu (24gb VRAM). So I decided to go with low-rank updates, as the gradient computation would have otherwise required too much VRAM. Second, I went with _whisper-turbo_ for the negative log likelihood scoring, rather than _whisper-large_. Probably, this leads to markedly lower-quality NLL rewards but such is life. Third, I use Nvidia _parakeet_ for the word error rate computation. It is simply faster, and I feel less finicky and more stable. All whisper models are susceptible to "freakouts" (hallucination from silence, repetitive loops), especially when dealing with noisy audio, and this tended to destabilize the reward signal.

Related to the decision to switch from whisper to parakeet, it seemed that the word error rate dominated the reward signal, i.e. had much higher variance than the other components. So I switched to character error rate, which is a little smoother, and down-weighted it as well. Additionally I did gradient accumulation. Before taking these steps, the train set actually diverged i.e. the reward got progressively worse.

So, here is some synthesized speech, after RL'ing for about 1 hour:

{% include embed-audio.html src="/assets/audio/whispering.wav" %}

As a reminder, I am optimizing for _"a man whispering"_. To me, the audio demonstrates that the model has _kind of_ got the gist of the concept, maybe even taking it too far. The audio has an unfortunate reverb artefact which is either a random apparition or some side effect learning to whisper; I need to check if there is any reward that could promote a "clean" signal. 

Some directions to go might include the following: generating a range of speaker styles by baking the description of the style into the prompt, then calculating the CLAP reward for each item based on its corresponding style description. I'm also not sure how good the CLAP model really is as a guidance model. During training that part of the reward never went above about 25% similarity. Maybe there was a direct tradeoff between NLL and the CLAP score. Or, perhaps a more powerful audio embedding model, such as [nvidia/omni-embed-nemotron-3b](https://huggingface.co/nvidia/omni-embed-nemotron-3b?linkId=100000387358775) (?) gives a better signal. But then I'd need to move this to a larger gpu setup (which would be sensible anyway). Moving to a bigger base model or full-rank updates may also be fruitful.