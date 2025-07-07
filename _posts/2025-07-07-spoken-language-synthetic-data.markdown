---
layout: post
title:  "Spoken Language Entity Linking: Synthetic Data"
date:   2025-07-07 13:01:09 +0200
categories: 
---

Let's take a look at the synthetic data for the spoken language entity linking system. We'll need a list of target entities, audio of users requesting those entities, and transcriptions of that audio.

We can download all open street map data for our countries of interest via [Geofabrik](geofabrik.de). The data is a rather large protobuf file which can be converted by osmconvert into larger xml file. Attempts to efficiently parse this thing and extract street names via with a streaming xml parser stumped Gemini code - presumably it only has access to the same 15 year old blog posts as I could find. In the end, one can just grep for the relevant parts of the xml structure, reducing the file size considerably, then do the streaming xml parsing to precisely extract the information.

We end up with about 500k unique street names, which we can split into train, validation and test sets.


# Text to speech

The diagram in the [first post](https://nrgow.github.io/2025/07/03/spoken-language-entity-linking.html) described the flow. [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) can generate from phonemes, so we provide the phonemes for German street names from a German phonemizer, Dutch streets from a Dutch phonemizer, and so on - all espeak-ng phoneme models.


{% include embed-audio.html src="/assets/audio/plommonvagen_sv.wav" %}

*"Navigate to Plommonvägen"*


{% include embed-audio.html src="/assets/audio/wustenwetzdorf_de.wav" %}

*"Navigate to Wüstenwetzdorf"*


{% include embed-audio.html src="/assets/audio/meerswal_nl.wav" %}

*"Navigate to Meerswal"*


There's plenty of other things that could be done, like audio augmentations, considering alternative TTS providers. But there is enough to work with here for the moment. I sample 50k street names to generate audio for.


# Speech to text

For transcribing our synthetic data, let's start with *whisper-turbo* - this is a more compact version of a multilingual ASR model, which in my experience can sometimes do surprisingly well with code-switched queries.

As an alternative, we can look at a more recent model: Parakeet (*nvidia/parakeet-tdt-0.6b-v2*) is and performant model from nvidia. It has the benefit of being ~20x faster than whisper-turbo. It currently second on the [open ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). It's English-only. Despite this project being about code-switching, one of the secret premises of this project is that we might even not need to know much about other languages, at least to do this somewhat artificial entity linking task. That's if all goes well.

The character error rate over the entities is pretty similar for parakeet vs. whisper-turbo (43%).

## Multimodal LLMs

Some new multimodal LLMs have been released recently, most notably phi-4-multimodal and gemma-3n. These accept speech audio or text (images as well) as input. Why might such a model be interesting for a simple transcription task? Incorporation of a text-based prompt potentially allows for zero-shot or few-shot customization of the ASR component without requiring finetuning. Let's see if that works with Gemma 3n *google/gemma-3n-E2B-it*. 

### Prompt help 1

All queries have the form "navigate to <street>", which I use to then extract the street name via regex. I already have to Gemma give a little help via the system prompt.

> "You are a helpful assistant who can transcribe audio. Users typically issue commands _in the imperative voice_."

The results are pretty bad. Character error rate is 57%. The transcriptions are not always phonetically plausible, rather making phonetically poorly-grounded semantic jumps to other entiti es, but we're not here to judge ASR models. We're trying to train a model that can correct for ASR errors, so the transcription may be valuable for that.


### Prompt help 2

As an experiment, let's give Gemma some more help with an enhanced task prompt:

> Transcribe this audio file which has this structure: 'Navigate to <german/dutch/swedish street name>'

Things start to look better. Character error rate decreases to 46%: worse than parakeet or whisper but closing the gap. In less than one percent of cases, it takes the prompt a little bit too literally, and outputs "_navigate to <german/dutch/swedish street name>_".

This example may be a little contrived, as one might rarely be able to give such a big hint, but it nicely demonstrates the flexibility and responsiveness of a multimodal model to enhanced context. In a real system, such context engineering could go in the direction of personalization and memory, or guidelines for particular terminology.


## Interim character error-rate evaluation

We can measure the ASR on our synthetic data, taking the character error rate of just the entity name. This evaluation is just for demonstration - maybe it is roughly correlated with end to end task performance.


|     |   parakeet |   whisper-turbo |   gemma_simpleprompt |   gemma_hintprompt    |
|-----|------------|-----------------|----------------------|-----------------------|
| cer |   0.43     |        0.43     |             0.57     |              0.46     |




## Transcription examples


| entity                  | parakeet               | whisper-turbo         | gemma_simpleprompt     | gemma_hintprompt        |
|-------------------------|------------------------|-----------------------|------------------------|-------------------------|
| Kölvägen                | Schellwegan            | Shellvegan            | Shelvegen              | Shelvegan               |
| Pfarrer-Held-Straße     | Pfaterhilshthasa       | Pfaderhildstädt Hasse | Powerchords to Hassa   | Pfahlstr                |
| Park de Wervelaan       | Parc de Vervelin       | Parc de Vervalan      | Park de Verelan        | Park de Verellen        |
| Untere Pfeifermühle     | Unterup Phi formula    | Unterupfi Formula     | Unter a five formula   | Unter der Pfeiferstraße |
| Posses väg              | Pausus Veg             | Posis Veg             | Passus Veg             | Passusweg               |
| Am Springgarten         | Mspttingaten           | Amschptingaten        | Amsterdam              | amshiptingaten          |
| Broängsvägen            | Brunswagen             | Bruinswagen           | Braunschweig           | Braunschweig            |
| Alander Weg             | Aulender Vake          | All under VAKE        | all under wake         | Allenderwiek            |
| Im Alengarten           | him all in Gatson      | Im Allenghattan       | M All in Gaten         | am Allingaten           |
| Nordbrink               | NodPink                | Nodbdink              | Notepad                | Nordpink                |
| Siegelhof               | Ziegelhoof             | Ziegelhof             | Ziegelhof              | Zeughof                 |
| Im Eisenfeld            | him Eisenfelt          | M. Eisenfeld          | Imfeld                 | Immenfeld               |


# Human control audio

It's pretty fast to just say things, so collecting a bit of real data for validating things at the end is very doable. _If only there was a nice tool to handle keep track of all the wav files you need to generate_. Gemini code can easily cook one up in 20 minutes (that's including a game of Minesweeper intermediate I played in between - really it was closer to 2 minutes).

Wielding this new tool, I can quickly generate 50 test set queries, and 50 validation set queries, which I'll evaluate on the end to end system later.

![Let's generate some human data too]({{ "/assets/images/data-collection-tool.png" | relative_url }})

*Let's generate some human data too*


With some transcriptions in hand, we can now start to train the embedding model.