---
layout: post
title:  "Code-Switched Spoken Language Entity Linking: Intro"
date:   2025-07-03 23:40:42 +0200
categories: 
---

Let's say you're on holiday with your voice assistant, and you want to get to _Bolzeschachtstraße_. You fire off a wakeup word, recording starts, you say: "_Navigate to Bolzeschachtstraße_". Your voice assistant hears "_Baltzsakstasa_". Another day, another place, this time _Zum Hallerbach_. Your voice assistant hears "_Tsamhalaba_". No results.

![Can we fix this?]({{ "/assets/images/typical_scenario.png" | relative_url }}){:width="40%"}

*Can we fix this?*


What is to blame? Clearly, speech recognition has a problem. It's great if you can fix it. On the other hand, a sufficiently smart voice assistant really *should* be able to guess, _even from an severely distorted transcription_, what is really intended, as long as the distortion is not totally wild.

The following blogposts will demonstrate a simple system that can take such garbled transcriptions from speech recognition and offer more accurate alternatives.

"Code-switched" means that the user will be speaking English, but will be requesting _non-English_ entities. To keep things compact, I will define the universe of entities as being: German, Dutch and Swedish street names. A large enough universe to make the task challenging. I may add more as time goes on. "Spoken language" means that the input to the system is speech audio, to be transcribed by an ASR module.

The core of this system will be a text-based phonetic embedding model. The similarity function the embedding model should try to capture is this: "string _x_ should be close to string _y_ if _x_ is a likely transcription from an ASR system of someone speaking _y_". So, phonetics is the main invariance that should be captured, but text normalization is another, as well as the output quirks of specific ASR models. The transformer backbone of the model will by a **byt5** encoder. I'll also consider a (not so?) silly baseline model based on _character ngrams_.

When people think of phonetic similarity, they may think of things like levenshtein distance. Unfortunately this won't scale into a retrieval system, so we'll have to use a sparse or dense search index. The character ngram approach might be considered an approximation of edit distance. Out of the box, it already seems pretty weak because it doesn't respect any of the invariances we are interested in. Maybe we can overcome this. In any case, there is, to my knowledge, no plug-and-play phonetic embedding model out there, so I'll have to build this out.

I'll take a look at a few different ASR models, phonetic embedding training strategies, and hybrid search & reranking approaches. I'll use _Qdrant_ as the vector search engine. For ASR, I will first consider **openai/whisper-large-v3-turbo** and **nvidia/parakeet-tdt-0.6b-v2**. But maybe some more recent multimodal LLMs which can accept audio input, like **google/gemma-3n-E2B-it** and **microsoft/phi-4-instruct**.

![Here's how it will all look]({{ "/assets/images/embedding-training-pipeline.drawio.png" | relative_url }})  

*Here's how the embedding model will be trained*

The training data for the embedding model will come from **hexgrad/Kokoro-82M** text-to-speech. It has the advantage that you can synthesize from phonemes. The idea is to phonemize the street name part of the TTS queries using a locale-appropriate phonemizer, to approximate an English speaker who can roughly pronounce the non-English street name. If one of the good commercial providers can do code switching beter than this, I might spend O(€10) on that too.