---
layout: post
title:  'Step-Audio-R1: a 33B parameter audio reasoning model'
date:   2025-12-05 09:30:00 +0100
categories: 
---

In previous blogposts I examined several ASR models for code switched entity linking. In that scenario, I could show that small models could perform well, especially in combination with a custom dense phonetic search index. 

What about extremely big speech models? One such recent model is [Step-Audio-R1](https://huggingface.co/stepfun-ai/Step-Audio-R1), with 33B parameters, trained for general audio understanding, incorporating a multimodal reasoning capability.

"_Reasoning over audio, how does that work?_", you may ask.

The model trains with a _modality-grounded reasoning distillation_. It starts with a text-only reasoner (Qwen2.5 32B), then progressively self-distills to prefer reasoning chains grounded in acoustic tokens, which come from the Qwen2 audio encoder. You can find a full description in [the paper](https://arxiv.org/abs/2511.15848).

Intuitively, here's what I would expect from a reasoning model for transcription of difficult audios:
1. diverse candidate generation based on the the rough shape of the sounds in the audio. 
2. hypothesizing about the most likely transcription based on some mixture of contextual, language-model and acoustic likelihood.

I would then expect the benefits of a reasoning model, compared to a traditional n-best hypothesis generation + reranking pipeline, to be in smarter types of hypothesis scoring. The elements of reasoning would then be phonetic similarities, knowledge of the distribution of pronuniciation e.g. _"English spoken in German accent ⇒ difficult named entities more likely to be German"_, this kind of thing.

Let's try it on the same personally generated eval set I used in previous posts, 50 code switched queries of the form "_Navigate to `<german_street>`_".

I ran this on a runpod A100 server. Inference can take anywhere between 10 seconds and two minutes.

Even when explicitly asked to begin its response with reasoning tags, the model occasionally immediately stops its reasoning efforts and delivers a transcription. One can "force" reasoning by prepending not just the start reasoning tag `<think>`, but the apparently universal start-reasoning word "_Okay_".

```python
messages = [
    {
        "role": "system", 
        # add contextual info
        "content": "You are an expert audio analysis and speech to text system. Think about what you hear, then when you are certain, output TRANSCRIPTION: and the transcription. User language: English. User location: Germany."
    },
    {
        "role": "human", 
        "content": [{"type": "audio", "audio": audio_path}]
    },
    {
        "role": "assistant", 
        "content": "<think>Okay,\n",  # force reasoning
        "eot": False
    },
]
```
*The prompt sent to the model including forced reasoning*


Additionally, it can benefit from extra information to reason over. In this case I give it contextual information (i.e. a hint) in the system prompt.

What performance do we get out of this behemoth?

| retrieval_model   |   k |   gemma-3n-2b-p1 |   parakeet |   **stepfunr1** |   voxtral-mini |   whisper-turbo |
|-------------------|-----|------------------|------------|-------------|----------------|-----------------|
| dense-v1          |   1 |              0.5 |       0.64 |        **0.42** |           0.56 |            0.74 |
| dense-v1          |  10 |             0.78 |       0.84 |        **0.6**  |           0.74 |            0.92 |
| dense-v1          |  40 |             0.82 |       0.92 |        **0.72** |           0.84 |            0.92 |
*higher is better*


So it doesn't perform so well, under this pretty limited evaluation.

Here is a typical reasoning trace:

> `<think>`  I need to figure out the correct transcription for the audio provided. Let me start by listening to the audio again carefully.
> The user mentioned that the language is English, but the location is Germany. Hmm, that's a bit confusing. Maybe the speaker is an English speaker in Germany, or perhaps there's a mix of languages. Wait, the transcription example given earlier had a German phrase: "Ist Landische Strasse". So maybe the audio is in German, but the user specified English by mistake.
> Wait, the user's instruction says: "User language: English. User location: Germany." But the example transcription they provided earlier was in German. Let me check the previous example. Oh, in the problem statement, the user provided an example where the audio is in German ("Ist Landische Strasse"), but the user language was set to English. Maybe there's a misunderstanding here.
> Wait, looking back at the current query: the user is giving an audio file, and the task is to transcribe it. The user specified "User language: English. User location: Germany." But the actual audio might be in German. Let me check the audio again.
> The audio says: "Navigate to East Landische Strasse". Wait, "East Landische Strasse" – "Landische" is a German word, meaning "of the country" or "rural". "Strasse" is German for "street". So the phrase is a mix: "Navigate to East Landische Strasse". But "East" is English, while "Landische Strasse" is German. Wait, but "East" in German would be "Ost", so maybe it's a mistranslation or a mix-up.
> Wait, perhaps the correct phrase is "Eisländische Straße", which would be "Icelandic Street" in German. Because "Eisländische" is the German adjective for "Icelandic". So "Navigate to Eisländische Straße" would be "Navigate to Icelandic Street". But the user wrote "East Landische Strasse" in the problem statement. Maybe that's a typo.
> Wait, in the problem statement, the user wrote: "Navigate to East Landische Strasse". But "East" in German is "Ost", so "Ostlandische Straße" would be "Eastern Street". But "Landische" alone isn't a standard German word. Wait, "landisch" in German can mean "of the land" or "rural", but "Eisländische" is "Icelandic".
> Ah! Maybe it's a misspelling. If the audio says "Eisländische Straße", which is "Icelandic Street" in German, then the correct transcription would be "Navigate to Eisländische Straße". But the user wrote "East Landische Strasse" which might be a mishearing or typo.
> Wait, let's break it down. The audio is in German, but the command "Navigate to" is in English. So the speaker is using English for the command and German for the street name. That's common in some contexts, like GPS systems in Germany might use English commands with German street names.
> So the phrase is: "Navigate to [German street name]".
> The street name in the audio is "East Landische Strasse". But "East Landische" – "East" is English, "Landische" is German. Wait, maybe it's "Östliche Straße", which means "Eastern Street". Because "östliche" is the German adjective for "eastern". So "Östliche Straße" would be "Eastern Street". But the user wrote "East Landische", which might be a mishearing of "Östliche".
> Alternatively, "Eisländische Straße" is "Icelandic Street". Let's check the pronunciation. "Eis" sounds like "ice", "ländische" – maybe the speaker said "Eisländische" but it was misheard as "East Landische".
> Given that the user is in Germany, it's more likely to be a German street name. So "Navigate to Eisländische Straße" would make sense as "Navigate to Icelandic Street".
> But the user's example in the problem statement says: "Navigate to East Landische Strasse". Maybe that's a typo. Let me listen to the audio again.
> Wait, since I can't actually hear the audio, I have to go by the information given. The user provided the audio file, but as a text-based model, I can't access it. Wait, no—the user is simulating the scenario where I have to transcribe the audio. Wait, in the problem setup, the user is providing the audio content as part of the query. Wait, looking back:
> The user's message is: "You are an expert audio analysis and speech to text system. Think about what you hear, then when you are certain, output TRANSCRIPTION: and the transcription. User language: English. User location: Germany."
> Then the user provides an audio file (in the context of the problem), but since this is a text-based interaction, the actual audio content is presumably embedded in the problem. Wait, perhaps in the actual scenario, the audio content is "Navigate to East Landische Strasse", but given that the location is Germany, it's more likely to be a German phrase.
> Wait, the correct German for "Icelandic Street" is "Eisländische Straße". The word "Eisländische" is pronounced with a 'y' sound, similar to "ice-lend-ee-shuh". If someone mishears "Eisländische" as "East Landische", that could happen because "Eis" sounds like "ice", which might be misheard as "East" (though phonetically different). Alternatively, "Östliche" (eastern) is pronounced with an umlaut, which might be misheard as "East Landische".
> Given that "Straße" is German for "street", the correct transcription should include the German street name. So "Navigate to Eisländische Straße" would be the correct phrase, meaning "Navigate to Icelandic Street".
> But the user's example in the problem statement says "East Landische Strasse". Maybe that's a typo, and the intended correct phrase is "Eisländische Straße".
> Alternatively, if the audio says "Östliche Straße", that would be "Eastern Street". Let's check: "Östliche" is "eastern", so "Östliche Straße" is "Eastern Street".
> Given the user's location is Germany, it's plausible. But which is more likely: "Eisländische" (Icelandic) or "Östliche" (Eastern)?
> Without hearing the audio, it's tricky. But given common street names, "Östliche Straße" (Eastern Street) is more common than "Eisländische Straße" (Icelandic Street), though both are possible.
> Wait, but "Eisländische" is the correct adjective for Icelandic. Let's think about pronunciation. "Eisländische" is pronounced "Ice-lend-ee-shuh", while "Östliche" is "Uhr-shtee-shuh" (with an umlaut on ö).
> If the speaker has a German accent, "Eisländische" might be misheard as "East Landische" if the 's' and 'l' are slurred. Alternatively, if the speaker is non-native, they might say "East Landische" incorrectly.
> But given the context of Germany, it's more likely to be a German phrase. So "Navigate to Östliche Straße" or "Navigate to Eisländische Straße".
> Another angle: the user's initial problem statement had a typo where they wrote "Ist Landische Strasse" instead of "Eisländische Straße". The word "Ist" is German for "is", but that doesn't fit. Wait, no—the example given in the problem was "Ist Landische Strasse", which is incorrect. Maybe it's a misspelling of "Eisländische Straße".
> Given all this, I think the correct transcription should be "Navigate to Eisländische Straße", assuming the audio says that. But the user's example says "East Landische Strasse", which is likely a misspelling of "Eisländische Straße".
> Alternatively, if the audio says "East Landische Strasse" as written, but that doesn't make much sense in German. "East" is English, "Landische" isn't a standard German word. Unless it's a specific street name that's a mix, but that's unlikely.
> Wait, "Landische" in German can be a misspelling of "ländische", which is an adjective form. "Ländische Straße" would be "rural street" or "country street". But "East Landische" doesn't parse.
> Alternatively, maybe it's "Ostlandische Straße", where "Ostland" refers to "eastern land", but that's not a standard term.
> Given the confusion, but considering common mistakes and the German context, I think the intended phrase is "Eisländische Straße", so the transcription should be "Navigate to Eisländische Straße".
> But the user's example in the problem statement wrote "East Landische Strasse", which might be a typo for "Eisländische". So the correct transcription, correcting the typo, would be "Navigate to Eisländische Straße".
> Alternatively, if the audio actually says "East Landische Strasse" as in English words, but that seems unlikely in a German context.
> Wait, another possibility: the speaker is using English for "Navigate to" and then a German street name, but mispronounced. So "East" is actually "Eis" (ice), and "Landische" is "ländische".
> So "Navigate to Eis ländische Straße" → "Navigate to Eisländische Straße".
> Yes, that makes sense. The speaker says "Eis" (ice) followed by "ländische Straße", but it's run together as "Eisländische Straße".
> Therefore, the correct transcription is "Navigate to Eisländische Straße".
> But the user's example had "East Landische", which is a mishearing of "Eisländische".
> So the final answer should be "Navigate to Eisländische Straße".
> `</think>`
> TRANSCRIPTION: Navigate to Eisländische Straße


This is one of the longer reasoning chains, but others are pretty similar. Some parts are pretty bizarre, suggesting that the model hasn't quite fully internalized the nature of its own circumstances. The reasoning starts off with a guess of the transcription, then drifts off to wondering about whether "the user" made a typo, or whether the user "wrote" this or that spelling of the street names. The model also claims, repeatedly, that it cannot hear the audio. So, at least for this use case, I fear the audio and text modalities have not been properly fused, or rather that the modality distillation still retains vestiges of the text-only reasoning initial model. The code-switched setup also causes some consternation for the model.

Anyway this setup is probably out-of-distribution and not doing justice to the capabilities of the model. I was simply curious what an audio reasoning model might be capable of reasoning about.

Here are the steps I needed to take to run this beast on the A100 instance, including minor changes compared to the model README.

```bash
git clone https://github.com/stepfun-ai/vllm.git
cd vllm
python3 -m venv .venv
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 pip install -e .
git checkout step-audio2-mini
pip install librosa hf_transfer
HUGGING_FACE_HUB_TOKEN=XXX hf download stepfun-ai/Step-Audio-R1 --local-dir /workspace/Step-Audio-R1
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/Step-Audio-R1 \    
    --served-model-name Step-Audio-R1 \
    --port 9999 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-log-requests \
    --interleave-mm-strings \
    --chat-template ... # as in the repo
```