# drum_along: design notes and known limits

Living notes on decisions, open questions, and what v1 does not do yet.

## The VST3 plug-in

Build: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` then
`cmake --build build --config Release`. The bundle lands in
`build/VST3/Release/drum_along.vst3`, with the weights and styles inside it
(`Contents/Resources/weights/`). Steinberg's validator runs after every build.

Install: copy `drum_along.vst3` (the whole folder) into
`C:\Program Files\Common Files\VST3` (admin) or
`%LOCALAPPDATA%\Programs\Common\VST3` (per user), then rescan in your DAW.

One bundle, two plug-ins:

| Plug-in | Where it goes | Use in |
|---|---|---|
| drum_along (FX) | On the band's audio track. Audio passes through, drum MIDI comes out. | Reaper, Bitwig, Cubase, Studio One |
| drum_along (Instrument) | On a MIDI track. Hears the band through its sidechain input. Audio out is silent. | Ableton Live (and any host) |

Ableton Live setup (Live only routes plug-in MIDI from instrument slots):
1. MIDI track A: insert **drum_along (Instrument)**. In its sidechain / audio
   input, choose the track with your band (or your live input).
2. MIDI track B: your drum instrument (Kontakt, Drum Rack...). Set its
   *MIDI From* to track A, then pick drum_along in the second chooser.
3. Arm/monitor track B. Live merges MIDI channels between tracks, which is
   fine here: all drum notes are on one channel.

Parameters:

| Parameter | What it does |
|---|---|
| Play | Off mutes the drums but keeps listening, so it re-enters in time. |
| Style | Rock, Alt Rock, Pop, Jazz. Switches on the next sixteenth without losing the beat. |
| Timing Offset | -50 to +50 ms. Negative plays earlier, to cancel your drum instrument's own latency. |
| MIDI Channel | 1 to 16, default 10 (General MIDI drums). |

Notes are General MIDI: kick 36, snare 38, closed hat 42, pedal hat 44, open
hat 46, crash 49, ride 51, toms 50 / 47 / 45. Most Kontakt drum libraries
accept GM mapping or have a GM preset.

Testing: `vst3_check` loads the built bundle the way a DAW does and plays a
WAV through it. Both plug-in classes produce MIDI identical, event for event,
to the `wav2mid` harness, so every offline evaluation is an evaluation of the
real plug-in.

## How the drummer is built

- Two kinds of weights. `weights/beatnet.json` is trained: about 510k weights
  that hear the beat, the downbeat and whether a drummer would play.
  `weights/drummer.json` is written by hand and tuned on the Slakh validation
  songs: how the drummer acts on what it hears.
- The hand-written starting values live in `core/src/WeightProgrammer.cpp`,
  with the reasoning next to each number. `weights/drummer.json` has moved on
  from them, so `wav2mid --dump-weights` must not overwrite it.
- The engine loads the file at startup, so editing it changes the drummer
  without recompiling. Rebuild only to copy it into the plug-in bundle. Run
  CMake configure again if `weights/beatnet.json` is new, or the bundle ships
  without it.
- The big front-end tables (DFT rows, mel and chroma filters) are not in the
  file. They are pure math derived from the file's ranges for the host
  sample rate.
- No BPM is ever passed in. Tempo, beat phase and the downbeat are inferred
  from the audio stream alone.

## Leadership: who follows whom

In a real band the drummer and the players are coupled clocks, but the band
mostly follows the drummer. `phase.leadership` (0..1, default 0.5) scales how
fast the drummer bends toward the band, both in phase and in tempo:

- 0: chases the band closely, like a follower.
- 1: holds its own time, like a metronome that drifts only slowly.

With a pre-recorded track the band never follows back, so very high
leadership makes the drummer slow to correct when it is wrong. For live
players, high leadership is the point: they lock to the drums.

## Tempo changes inside a song

Current behaviour (v1):

- Small drifts (a band speeding up a few percent) are followed continuously.
- A tempo jump, such as 60 to 80 BPM, is only accepted once the old tempo
  has stopped repeating in the music and the new one has clearly won for
  3 seconds. In practice that takes about 5 to 7 seconds, during which the
  drummer keeps playing the old tempo.
- For 12 seconds after the first lock there is a settling window, where an
  early wrong count can be replaced more easily.
- A real stop in the band reopens that window, because a stop is when a new
  tempo is most likely. A stop means loudness has collapsed: below 0.6 of the
  last 3 seconds, on the log scale, for a beat. Reading missing attacks instead
  misfired, because a dense wall of guitars has almost no attacks either.
  When the band returns, half of the tempo memory is dropped.

Measured with `scripts/eval_pauses.py`, on validation songs with stops and
tempo changes cut in: 0.04 stumbles per stop, and every one of 24 tempo changes
found, 7 seconds after the band returns (median).

Deliberate trade-off: stability beats agility. Flipping tempo in a quiet
breakdown sounds far worse than being late to a real tempo change.

Ideas for later:

- Lay out (stop playing) as soon as the current tempo stops fitting, instead
  of playing the old tempo until the switch. Silence is better than wrong.
- Detect a tempo change from the phase error drifting consistently in one
  direction, which happens within a bar, long before the autocorrelation
  memory (4 s) catches up.
- A "tempo map" mode for songs with known changes.

## When the real drummer is silent

Some songs have no drums in the intro, a breakdown or the outro. The learned
"drummer would play" output decides, smoothed over 2 seconds. Below
`beatNet.layOutBelow` the drummer sits out; it comes back above
`beatNet.layInAbove`, always on a bar line, with a crash. The two thresholds
are a real trade-off, measured on the 39 validation songs with long intros:

| Out / in | Comes in early by (median) | Mean start | Sits out real drums |
|---|---|---|---|
| Off | 38 s | 9 s | 0% |
| 0.3 / 0.5 | 22 s | 13 s | 1% |
| **0.5 / 0.7 (shipped)** | 19 s | 15 s | 2% |
| 0.7 / 0.85 | 4 s | 19 s | 4% |

So intros are better, not solved.

The training labels come from each song's own MIDI drum part. Drums play 88% of
the time in Slakh, so that head is trained with the silent frames weighted up.

## Staying steady

Listening found what the scores missed, twice:

- The clock used to jump half a beat whenever a syncopated riff beat the true
  beat for a second, and jump back seconds later. Half of those jumps made
  things worse. Now a different phase must keep winning for 16 seconds, or for
  8 seconds in a settling window. `pause` and `lurch` in `eval_dataset.py`
  count these moves, because beat F barely notices them.
- The pattern used to follow intensity live, so it changed on 104 of 130 bar
  lines in one song. Now it is chosen on a 4-bar line and only for a real change
  of intensity. Velocity still follows every step.

## Results

Whole drummer on 119 Slakh test songs it never saw, against MIDI truth:

| | Hand-written only | Shipped |
|---|---|---|
| Beat F | 0.561 | 0.813 |
| On the beat | 59% | 84% |
| On the "&" | 27% | 9% |
| "1" on the real 1 | 25% | 69% |
| Snare on the right two beats | 38% | 78% |
| Right tempo | 63% | 85% |
| Stumbles per minute | 4.5 | 0.1 |

Validation scored about the same (F 0.822), so tuning did not over-fit the
split. Tuning on a 120-song subset did flatter that subset (F 0.863), so judge
on the whole split.

Known weak spots:

- Slakh has no vocals, so real songs with singing are a harder test than
  these numbers. Beat versus "&" is still the weakest decision on real records.
- With the real drums removed, a groove half a beat from the real snare can
  still sound right, so listening and beat scores can disagree.

## Data and licensing

The shipped weights must be commercially usable, so every dataset is checked.

| Source | License | Use here |
|---|---|---|
| Slakh2100 (redux) | CC BY 4.0, commercial use OK with credit | **Training** the learned layer |
| BabySlakh | CC BY 4.0 | Pipeline smoke tests only |
| Personal test songs | Private copies, never in this repo | Listening tests only, never training |
| MUSDB18-HQ, MedleyDB, MTG-Jamendo, MoisesDB | Non-commercial | Not used |
| OnAir | CC BY-SA 4.0 | Not used (share-alike; evaluation at most) |
| Demucs pretrained models | Maintainer says research use only | Personal test splits only, never to build training data |

**Required attribution** (CC BY 4.0) for any build that ships weights/beatnet.json:

> Slakh2100: E. Manilow, G. Wichern, P. Seetharaman, J. Le Roux, "Cutting Music
> Source Separation Some Slakh: A Dataset to Study the Impact of Training Data
> Quality and Quantity", WASPAA 2019. CC BY 4.0. https://zenodo.org/records/4599666

Slakh's audio is rendered from MIDI with virtual instruments, so it is clean
and has no vocals. Bar lines are checked against each song's own drum part
(`scripts/diag/grid_census.py`): songs whose snare does not sit on the MIDI's 2
and 4 do not train the downbeat. Training adds a third, augmented version of every backing
(random EQ tilt, saturation, room reverb, noise, level) so the layer does not
over-fit to clean synthesis. Vocals remain a known gap: real songs with
singing are only checked by listening to personal test songs.

## Learned layer pipeline

1. `scripts/data/get_slakh.cmd` downloads the official archive with a
   resumable `curl -C -` (104 GB, single connection), then runs
   `scripts/data/slakh_stream_build.py` on the local file. That keeps only the
   non-drum stems and writes two drumless backings per song plus its MIDI.
   Allow about 215 GB at the peak: archive, about 65 GB of temporary stems, and
   about 45 GB of backings. Result: 1,710 songs (1,289 train, 270 validation,
   151 test), 118 hours.
2. `scripts/train/make_features.py` runs the C++ harness with `--features` on
   each backing (plus the augmented variant), so training sees exactly the
   numbers the plugin computes. Labels come from the song's own MIDI: beats,
   downbeats, and where the drums play.
3. `scripts/train/train_beatnet.py` trains the causal TCN and exports
   `weights/beatnet.json`. Shipped: 128 channels, dilations 1 to 512 (20 seconds
   of memory), about 510k weights, 100 epochs. Sizes tried: 32 and 48 channels
   with 5 seconds of memory, 64, 96 and 128 with 20 seconds. Memory is what
   lifted the downbeat. A parity test confirmed the C++ and PyTorch outputs
   agree to 5e-5.
4. `scripts/eval_dataset.py` scores the whole drummer on the validation split
   against MIDI truth. This is the loop for hand-editing `weights/drummer.json`.
5. `scripts/tune_weights.py` runs that loop one weight at a time: it tries each
   candidate value on the same songs and keeps the winners.

Long jobs run next to other heavy work on this machine, so the trainer reads
feature slices from disk instead of memory-mapping them: its RAM stays flat and
the OS file cache does the speeding up.

## Test material policy

- Personal test songs are listening material only, never used to set weights
  and never committed.
- Hand-tuning uses the Slakh validation split. The Slakh test split was used
  once, for the results above. Tuning against it would make that score
  meaningless.
- To judge a personal song, `scripts/diag/pocket.py` shows where the real snare
  lands on our bar.

## Cost

About 0.26 ms per 10 ms frame at 44.1 kHz (roughly 2.6% of one CPU core), of
which the learned layer is 0.12 ms. The two DFT layers are evaluated with an FFT: the same linear map as the
dense matrices (features agree to 3e-6, drum events identical), 3.4x faster.
