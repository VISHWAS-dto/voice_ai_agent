# Evaluation

Offline evaluation of the voice attribute inference model
(`app/inference/model.py`). Run it with:

```bash
python eval/run_eval.py                       # uses eval/datasets/manifest.csv
python eval/run_eval.py --json eval/reports/run.json   # + dump raw results
python eval/run_eval.py --help                # includes Common Voice how-to
```

It loads `AttributeInferencer` **directly** (not through the HTTP API — we
want to measure the model, not FastAPI), runs every clip in the manifest,
and prints:

- **gender accuracy** — % correct over clips where the model committed to
  `male`/`female`. The `unknown` (abstain) rate is reported *separately* so
  abstentions are never counted as right or wrong.
- **age-bracket accuracy** — same, exact-bracket match, abstain rate
  reported separately.
- **a calibration table** — predictions bucketed by the confidence the
  model reported (`0.5–0.6`, `0.6–0.7`, …) with the actual hit-rate inside
  each bucket. If the model says `0.9` a lot and is right 60% of the time,
  its confidence is not trustworthy and this is where you see it.

Results are also broken out for **verified-label clips only**, so a pile of
best-effort guessed labels can't quietly inflate the headline number.

## Honest limitations — read before quoting any number

- **This is a stub eval on a tiny, manually-curated set.** As shipped, the
  manifest has **one** clip (`sample.wav`). Even filled out to 10–20 clips
  it is an *anecdote*, not an evaluation: the confidence interval on an
  accuracy computed from ~15 samples is roughly ±25 points. Every number
  the script prints describes *those specific clips*, not "the model is X%
  accurate". Say that out loud in the README/video.
- **No class balance.** 12 men + 2 women in the manifest → a model that
  always guesses "male" scores 86% and you've learned nothing. The script
  does not enforce or check balance.
- **The bundled label is unverified** (`verified=no` in the manifest).
  `sample.wav` is the LDC93S1 TIMIT smoke-test sentence, commonly cited as
  a male speaker, but the label here hasn't been independently checked and
  the speaker's age is unpublished, so the age bracket is a guess. It is
  excluded from the "verified-only" numbers for that reason.

## Why not Mozilla Common Voice out of the box?

Common Voice is the natural source of age/gender-labeled speech, but every
current version on the Hugging Face Hub is a **gated dataset**: using it
needs a HF account, an access token, accepting the dataset licence, the
`datasets` package, and a multi-GB download. That was deliberately kept out
of the default path so `python eval/run_eval.py` runs with zero setup and
no account.

If you *do* have a token, `python eval/run_eval.py --help` prints the ~15
lines that pull ~20 labeled clips from `common_voice_17_0`, write them into
`eval/datasets/`, and append rows to `manifest.csv` (including the
CV-age-string → bracket mapping). Note that CV age/gender is **self-reported
and skews young + male**, so even a 20-clip pull is a spot check, not a
benchmark.

## Adding your own clips

Drop audio files anywhere (e.g. `eval/datasets/`) and add a row per clip to
`eval/datasets/manifest.csv`:

```
path,gender,age_bracket,verified,notes
eval/datasets/clip01.wav,female,31-45,yes,internal recording - age from HR
```

Columns and conventions are documented in the header comment of
`manifest.csv`. Relative paths resolve against the repo root. Any format
`ffmpeg` can read works. A clip that fails to decode or infer is logged and
skipped — one bad file never aborts the run.

## Files

- `run_eval.py` — the harness (loads the model, scores the manifest, prints
  the report). Metrics + calibration logic live in the same file.
- `datasets/manifest.csv` — checked in; the audio it points to is **not**
  (see `.gitignore`).
- `reports/` — `--json` output lands here; git-ignored.
