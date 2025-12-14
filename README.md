# Low-latency Streaming Speech-to-Text/Automatic Speech Recognition

## Goal
To reach speech to text with low latency faster than 100-150 msec, much faster than Whisper and other non-streaming automatic speech recognition (ASR).

## Keywords
Low latency, 100 msec, Speech-to-text, Automatic Speech Recognition, ASR, Conformer, Enformer, RNN-T, FPGA.

## Learning Path
1. **Load pretrained Emformer-RNN-T**: bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

2. **Real streaming loop**: mic callback → ring buffer → streaming FE → decoder.infer(..., state, hypothesis)

3. **Measure latency**: first token latency and compute time rolling p50/p95

4. **Tune knobs**:

- --beam-width 1 (lower latency)

- --commit-stability 1..6 (flicker vs responsiveness)

- --reset-hypothesis-on-silence (more “utterance-like”)

5. **Stress test**: --stress-cpu (and you’ll see dropped audio chunks if you can’t keep up)

## Run Examples
Low latency, more twitchy:
```
python streaming_rnnt.py --beam-width 1 --commit-stability 2 --device 2
```

More stable text, a bit more lag:
```
python streaming_rnnt.py --beam-width 4 --commit-stability 5 --device 2
```

Stress test:
```
python streaming_rnnt.py --stress-cpu --stress-intensity 0.85
```

## Installation
Create a virtual environment with Python 3.11.

```
pip install -r requirements.txt
```

## Block Diagram
```
Microphone (CoreAudio / mic HW)
          |
          | PCM samples @ 16 kHz
          v
Audio Capture Thread (sounddevice callback)
  - never block
  - drop if queue full
          |
          v
Audio Ring Buffer (bounded)
  - smooth callback jitter
  - prevent unbounded latency
          |
          v
Streaming Feature Extractor (log-mel)
  - framing
  - FFT
  - mel filterbank
  - hop = 10 ms
          |
          v
Chunk Assembler
  - fixed segment length
  - right-context overlap (lookahead)
          |
          v
Encoder (Axis 1): Emformer / Conformer (streaming)
  - cached acoustic state
  - bounded lookahead
          |
          v
Decoder (Axis 2): RNN-T
  - prediction network (implicit LM)
  - joint network
  - emits token or blank
          |
          v
Partial Stability / Commit Policy
  - commit after N stable steps
  - never revoke committed text
          |
          v
Output / UI
  - live partial text
  - committed transcript
```


## Debugging Help
```
python streaming_rnnt.py --beam-width 4 --commit-stability 5 --device 2 --debug-log --debug-every 10
```

## Test with MacOS microphone
```
python -m sounddevice
```
We are expected to see results
```
% python -m sounddevice
  0 iPhone Y Microphone, Core Audio (1 in, 0 out)
  1 BlackHole 2ch, Core Audio (2 in, 2 out)
> 2 MacBook Pro Microphone, Core Audio (1 in, 0 out)
< 3 MacBook Pro Speakers, Core Audio (0 in, 2 out)
  4 Microsoft Teams Audio, Core Audio (1 in, 1 out)
  5 rekordbox Aggregate Device, Core Audio (0 in, 2 out)
  6 ZoomAudioDevice, Core Audio (2 in, 2 out)
```

## Output Example
I played the introduction chapter of the audiobook "How Language Began" by Daniel L. Everett and the following is the result of the recognized text. 

```
% python streaming_rnnt.py --beam-width 1 --commit-stability 5 --device 2 --debug-log --debug-every 10

Using input device: MacBook Pro Microphone (id=2, default_sr=48000.0)

Listening... Ctrl+C to stop.

bundle=EMFORMER_RNNT_BASE_LIBRISPEECH sr=16000 segment=16 frames (0.160s) right_ctx=4 frames beam=1 commit_stability=5


[debug] step=10 rms=0.0023 tokens=1 text=''
 [introdu]
[debug] step=20 rms=0.0010 tokens=3 text='introduction'
introductionon]
[debug] step=30 rms=0.0115 tokens=3 text='introduction'
introduction [ in the beginning was]
[debug] step=40 rms=0.0142 tokens=7 text='introduction in the beginning was'
introduction in the beginning was the wordrd]
[debug] step=50 rms=0.0022 tokens=9 text='introduction in the beginning was the word'
introduction in the beginning was the word [ john chapter one]
[debug] step=60 rms=0.0061 tokens=12 text='introduction in the beginning was the word john chapter one'
introduction in the beginning was the word [ john chapter one first]
steps=62 dropped_audio=0 | first_token_ms p50=249.9 p95=249.9 n=1 | compute_ms p50=40.8 p95=75.6 n=62
introduction in the beginning was the word [ john chapter one first one]
[debug] step=70 rms=0.0007 tokens=14 text='introduction in the beginning was the word john chapter one first one'
introduction in the beginning was the word john chapter one first one [ no]
[debug] step=80 rms=0.0108 tokens=15 text='introduction in the beginning was the word john chapter one first one no'
introduction in the beginning was the word john chapter one first one no [ it was it]
[debug] step=90 rms=0.0036 tokens=19 text='introduction in the beginning was the word john chapter one first one no it was it then'
introduction in the beginning was the word john chapter one first one no [ it was it then everett]
[debug] step=100 rms=0.0016 tokens=23 text='introduction in the beginning was the word john chapter one first one no it was it then everett'
introduction in the beginning was the word john chapter one first one no it was it then everett [ it]
[debug] step=110 rms=0.0153 tokens=24 text='introduction in the beginning was the word john chapter one first one no it was it then everett it'
introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in]
[debug] step=120 rms=0.0063 tokens=32 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in ninet'
introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one]
steps=125 dropped_audio=0 | first_token_ms p50=249.9 p95=249.9 n=1 | compute_ms p50=41.3 p95=75.2 n=125
introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one along]
[debug] step=130 rms=0.0121 tokens=38 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that'
introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in]
[debug] step=140 rms=0.0166 tokens=42 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in'
introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a introduction in the beginning was the word john chapter one first one no it was it then everett [ it was a oduction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultuction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultryuction in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris]
[debug] step=150 rms=0.0072 tokens=50 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doub'
n in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morn in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morni in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morni in the beginning was the word john chapter one first one no it was it then everett [ it was a sultry mornin the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morningn the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morningn the beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in n beginning was the word john chapter one first one no it was it then everett [ it was a sultry morning in ninning was the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two]
[debug] step=160 rms=0.0119 tokens=55 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two'
inning was the word john chapter one first one no it was it then everett [ it was a sultry morning in ninetas the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen nineas the word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen nine word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one word john chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety onejohn chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one alongjohn chapter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one alongpter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred]
[debug] step=170 rms=0.0068 tokens=59 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred'
pter one first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gie first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio rivere first one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio riverrst one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in t one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m one no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s  no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o k no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o k no it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o k it was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in]
[debug] step=180 rms=0.0169 tokens=67 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a'
t was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king bt was it then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king bt then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris dot then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris dot then everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doen everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd severett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd severett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a single engine]
steps=188 dropped_audio=0 | first_token_ms p50=249.9 p95=249.9 n=1 | compute_ms p50=41.7 p95=74.8 n=188
everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a single engine]
[debug] step=190 rms=0.0076 tokens=70 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a single engine'
everett [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd stt [ it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some t it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two h it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two ha sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred ma sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred ma sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred mtry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles try morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles try morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a single engine from the nearest town]
[debug] step=200 rms=0.0027 tokens=75 text='introduction in the beginning was the word john chapter one first one no it was it then everett it was a sultry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a single engine from the nearest town'
try morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles try morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles ltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred milesltry morning in nineteen ninety one along that gio river in m s o king boris doubl'd some two hundred miles three hundred twenty kilometers in a single engine from the nearest town^C

Stopped.
steps=209 dropped_audio=0 | first_token_ms p50=249.9 p95=249.9 n=1 | compute_ms p50=41.6 p95=75.4 n=209
```