# Voice Attribute Inference Service

A small HTTP service that listens to a short audio clip of a caller and
returns a best-effort guess of the speaker's **gender** and **age
bracket**, plus a rating of how usable the audio was.

## 1. Overview

This service is a personalization helper for a logistics company's voice AI
call system. When a customer calls about a shipment, the voice agent can use
a coarse read of who is on the line — likely age band, likely gender — to
adjust tone, pacing, and script (e.g. slower and more formal for an older
caller, skipping app-based self-service prompts for someone who is unlikely
to use the app). It takes a `POST /analyze` with an audio clip and returns a
JSON body with `gender`, `age_bracket`, per-field confidence, an
`audio_quality` grade, and end-to-end processing time. It is explicitly a
*soft signal*: the predictions are noisy, the confidence numbers are
conservative, and the endpoint is built so that any failure inside it
degrades to "unknown" rather than breaking the call. It is not identity
verification and must never be used as such.

## 2. Setup

Requires **ffmpeg** (a system binary — the audio layer shells out to it over
pipes; it is not a pip package) and the Python dependencies in
`requirements.txt`. The acoustic model is ~1.2 GB of weights that download
once from the Hugging Face Hub into `.model_cache/` and are then reused.

### Local development

```bash
# 1. ffmpeg
brew install ffmpeg            # macOS
# sudo apt-get install -y ffmpeg   # Debian/Ubuntu

# 2. virtualenv + deps
python3 -m venv myenv
source myenv/bin/activate      # Windows: myenv\Scripts\activate
pip install -r requirements.txt

# webrtcvad still imports the legacy pkg_resources; on new setuptools you
# may need:  pip install "setuptools<81"

# 3. run the service (loads the model once on startup — first boot pays the
#    one-time weight download; subsequent boots load from .model_cache/)
uvicorn app.main:app --reload

# 4. smoke test
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/analyze -F "audio=@sample.wav" | python -m json.tool
```

The interactive API docs are at `http://localhost:8000/docs`.

### Docker

```bash
# Build (this bakes the ~1.2 GB of model weights into the image, so the
# build downloads them and takes a while; the layer is cached afterwards).
docker compose build

# Start, detached.
docker compose up -d

# Wait for it to report healthy — the container is only "healthy" once the
# model has finished loading.
docker compose ps

# Check it.
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/analyze -F "audio=@sample.wav"

# Logs (structured JSON on stdout).
docker compose logs -f api

# Stop. The model-cache named volume survives this, so `up` again does not
# re-download or re-seed the weights.
docker compose down
```

To rebuild from a completely clean slate (no cached image layers, no saved
weights):

```bash
docker compose down -v          # -v also drops the model-cache volume
docker compose build --no-cache
docker compose up -d
```

### Running the tests

```bash
source myenv/bin/activate
pytest -q                       # whole suite
pytest tests/test_quality.py -q # just the quality-gate unit tests
pytest tests/test_analyze.py -q # just the API integration tests
```

The integration tests (`tests/test_analyze.py`, `tests/test_reliability.py`)
boot the real FastAPI app through `TestClient`, which runs the startup
lifespan and loads the **real** model once for the module — they are not
mocked. They need `ffmpeg` on `PATH` and the weights already cached in
`.model_cache/`. `tests/test_quality.py` is pure and needs neither.

Current state of the suite: **38 passed, 1 skipped** (the skipped one is
`tests/test_schemas.py`, an untouched stub).

### Running the eval harness

```bash
source myenv/bin/activate
python eval/run_eval.py                                 # uses eval/datasets/manifest.csv
python eval/run_eval.py --json eval/reports/run.json    # also dump raw per-clip JSON
python eval/run_eval.py --help                          # prints a Common Voice how-to
```

`eval/run_eval.py` loads `AttributeInferencer` **directly** (not through the
HTTP API — the goal is to measure the model, not FastAPI), runs every clip
in `eval/datasets/manifest.csv`, and prints gender accuracy, age-bracket
accuracy (both with the abstain/"unknown" rate reported separately), and a
calibration table. As shipped the manifest has **one** clip (`sample.wav`);
see [§6](#6-known-limitations) and `eval/README.md` for why, and for how to
extend it. A clip that fails to decode or infer is logged and skipped; one
bad file never aborts the run.

## 3. API contract

### `POST /analyze`

**Request:** `multipart/form-data`.

| part         | type   | required | notes                                                        |
|--------------|--------|----------|-------------------------------------------------------------|
| `audio`      | file   | yes      | The clip. Any container/codec ffmpeg can read: wav, mp3, m4a, ogg/opus, webm, or raw 8 kHz G.711 mu-law/a-law. |
| `contact_id` | string | no       | Accepted for backward compatibility but **ignored** — a fresh uuid4 is minted per request so the value in the response and logs is unambiguous. |

**Response:** always `application/json`, always **HTTP 200** (see
[§4](#4-architecture--design-decisions) on why there is no error status).

| field                     | type              | meaning                                                                 |
|---------------------------|-------------------|------------------------------------------------------------------------|
| `contact_id`              | string (uuid4)    | Generated per request.                                                 |
| `gender.prediction`       | `male` \| `female` \| `unknown` | `unknown` when the model's top class is "child", on low-signal audio, or on any internal failure. |
| `gender.confidence`       | float `[0,1]`     | Softmax probability of the reported class. `0.0` when `prediction` is `unknown`. |
| `age_bracket.prediction`  | `18-30` \| `31-45` \| `46-60` \| `60+` \| `unknown` | `unknown` when the regressed age is < 18 or on failure. |
| `age_bracket.confidence`  | float `[0,1]`     | A **heuristic** interiority score (see [§4](#4-architecture--design-decisions)), roughly `[0.30, 0.65]`. **Not a calibrated probability.** `0.0` when `prediction` is `unknown`. |
| `processing_ms`           | int               | End-to-end wall-clock: read + decode + VAD + quality + inference.      |
| `audio_quality`           | `good` \| `degraded` \| `insufficient` | `insufficient` means the model was skipped and predictions are `unknown`. |

**Example:**

```bash
curl -s -X POST http://localhost:8000/analyze \
  -F "audio=@sample.wav" | python -m json.tool
```

```json
{
    "contact_id": "9f1c2b7e-3d4a-4e8f-a1b2-c3d4e5f60718",
    "gender": {
        "prediction": "female",
        "confidence": 0.94
    },
    "age_bracket": {
        "prediction": "31-45",
        "confidence": 0.58
    },
    "processing_ms": 214,
    "audio_quality": "good"
}
```

(The `contact_id` and predictions vary per call. On the bundled `sample.wav`
the model actually returns `gender: "unknown"` / `age_bracket: "18-30"` —
see [§6](#6-known-limitations).)

### `GET /health`

Liveness/readiness probe for orchestrators. No parameters.

```bash
curl -s http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

`model_loaded` is `true` only after the startup lifespan has finished
loading the ~1.2 GB model, so a load balancer can hold traffic until the
service can actually serve. `GET /healthz` is a backward-compatible alias
for the same check.

### `WS /ws/analyze`

A **bonus** streaming endpoint for progressive predictions as audio arrives
(`app/api/websocket.py`). It is a thin orchestration layer over the exact
same pipeline as `POST /analyze` — it does not re-implement decode / VAD /
quality / inference.

**Protocol:**

1. Client connects and sends **binary** messages: raw **16 kHz mono
   little-endian float32 PCM** (the canonical format `normalize_audio()`
   emits), any chunk size — chunks need not align to a window.
2. The server appends bytes to an in-memory rolling buffer. Every time
   **2 s** of audio has accumulated it detaches that window and runs
   `normalize_audio → run_vad → assess_quality → (skip if "insufficient")
   → AttributeInferencer.predict()`, then sends one **text** message: an
   `/analyze`-shaped JSON body plus two extra fields —
   - `window_index` (0-based) so the client can watch predictions evolve,
   - `best_so_far`, the running best (see below).
3. The client ends the stream by closing the socket, or by sending the
   text frame `"close"`. The server flushes any remaining partial buffer
   as one last window, sends `{"event": "closing", "reason": ...,
   "windows_processed": N, "best_so_far": {...}}`, and closes.

**Running "best" strategy:** the window with the **highest gender
confidence seen so far** wins (a plain argmax over windows), age carried
from that same window. Chosen over confidence-averaging because averaging
only makes sense while the same class keeps winning, so it needs per-class
bookkeeping and a policy for when the winning class flips mid-call — more
moving parts, more to get subtly wrong. Windows graded `insufficient`
(model skipped, confidence `0.0`) never displace a real prediction. See
the `_StreamSession.observe` docstring.

**Reliability & privacy (same rules as `/analyze`):**

- Each window runs in a worker thread under a 3 s budget; on overrun or a
  decode failure that window's message is all-`unknown` / `insufficient`
  rather than stalling or dropping the stream.
- A client disconnect is caught and turns into clean buffer cleanup — no
  crash, no traceback, one structured log line.
- **Nothing is ever written to disk.** Chunks live in an in-memory
  `bytearray` and are decoded by streaming into ffmpeg over a pipe.
- **No raw audio or raw bytes are logged** — one structured line per
  window with only `contact_id`, byte/sample *counts*, timing, the
  quality grade, and prediction *labels*.
- **Resource guards** so a client cannot hold the socket open or exhaust
  memory: max session duration (120 s), max cumulative received bytes,
  max window count (60), and a 30 s idle-receive timeout. Tripping any of
  them sends `{"event": "closing", "reason": ...}` and closes with code
  1008.

**Test client:** `scripts/test_websocket.py` decodes `sample.wav` once
(in memory, via `normalize_audio`), streams it in ~0.5 s chunks with a
short delay between sends to mimic real-time arrival, prints each incoming
prediction, and closes cleanly on EOF.

```bash
# with the service running (uvicorn app.main:app):
python scripts/test_websocket.py
python scripts/test_websocket.py --file path/to/other.wav
```

## 4. Architecture / design decisions

```
POST /analyze
   │  read bytes into memory (no disk, no tempfile)
   ▼
app/audio/ingest.py  normalize_audio()   ── ffmpeg over stdin/stdout ──▶ 16 kHz mono float32
   ▼
app/audio/ingest.py  run_vad()           ── webrtcvad, 30 ms frames ──▶ speech_ratio, durations
   ▼
app/audio/quality.py assess_quality()    ──▶ good | degraded | insufficient
   │
   ├─ insufficient ──▶ skip the model, return all-"unknown"
   ▼
app/inference/model.py  AttributeInferencer.predict()  ── one wav2vec2 forward pass ──▶ gender + age
   ▼
AnalyzeResponse (always HTTP 200)
```

### Why a wav2vec2 age/gender model (`audeering/wav2vec2-large-robust-24-ft-age-gender`)

The alternatives were classical hand-crafted acoustic features (openSMILE /
eGeMAPS feeding a small classifier) or a two-model pipeline (a separate
gender classifier and a separate age model).

We chose the single wav2vec2 model for two concrete reasons:

- **Robust pre-training matches the workload.** The backbone is
  wav2vec2-large-*robust*, pre-trained on a deliberately messy mix that
  includes Switchboard and Fisher *telephone* speech alongside read speech
  and noisy in-the-wild audio. A logistics call center is exactly that
  domain — 8 kHz codec audio, line noise, warehouse/truck background,
  cross-talk. A model fine-tuned only on clean studio speech tends to
  collapse toward one prediction on that input.
- **One forward pass, two heads = lower latency.** The model emits a pooled
  embedding once, then a regression head predicts age and a 3-class head
  predicts gender (child / female / male). We get both attributes for the
  cost of a single inference. A two-model approach doubles the forward-pass
  cost and the memory footprint for no accuracy gain here.

**The tradeoff we accepted:** openSMILE features are *interpretable* — you
can point at F0, jitter, shimmer, formant dispersion and say why a decision
went the way it did, and you can tune per-feature. A wav2vec2 embedding is
an opaque 1024-dim vector; when it is wrong, there is nothing to inspect.
For a soft personalization hint where latency and robustness to bad audio
matter more than explainability, that was the right trade. If this ever
needs to be audited or defended per-decision, the classical-features path
should be revisited.

### Age: raw regression → 4 brackets, and why the confidence is a heuristic

The age head outputs a single scalar `v` in `[0, 1]`; the model card's
convention is `age_years = v * 100`. `app/inference/model.py` (`_map_age`,
around lines 364–411) maps that continuous year onto the four contract
brackets using the brackets' own edges as cut points — nothing cleverer:

| predicted years | bracket   |
|-----------------|-----------|
| `[18, 30]`      | `18-30`   |
| `(30, 45]`      | `31-45`   |
| `(45, 60]`      | `46-60`   |
| `> 60`          | `60+`     |
| `< 18`          | `unknown` |

The boundary year falls in the *lower* bracket (30 → `18-30`), matching how
people say ranges ("early thirties" starts at 31). A regressed age below 18
returns `unknown` rather than being forced into `18-30`: the product only
cares about adult callers, and a sub-18 output usually means a childlike
voice, out-of-distribution audio, or a clip too degraded for the head to
commit — none of which should read as a confident `18-30`.

**The `age_bracket.confidence` value is not a calibrated posterior.** The
regression head emits one number and no distribution, so there is no honest
probability to report. What we surface instead is a deliberately modest
*interiority* proxy: how far the predicted year sits from the nearest edge
of its bracket, as a fraction of half the bracket width, squeezed into
`[0.30, 0.65]`. A prediction on a bracket edge scores ~0.30; one dead-center
scores ~0.65. It never approaches 1.0. The reason this matters:
near-silence and noise both regress toward a mid-bracket year, and without
this cap that would surface as a *high-confidence* age. Callers should read
this field as "roughly how interior to the bucket is the point estimate",
not as "the model is 65% sure".

The gender confidence *is* the softmax probability of the reported class —
that head is a real classifier — but note it is only reported for `male` /
`female`. A "child"-dominant output folds to `gender: "unknown"` with
confidence `0.0`, because the child probability is not a male/female
confidence and the API has no "child" value.

### The quality-gating pipeline, and why `insufficient` short-circuits

`normalize_audio()` produces a 16 kHz mono waveform; `run_vad()`
(`app/audio/ingest.py`) runs WebRTC VAD over it in 30 ms frames and returns
`speech_ratio` (VAD-detected speech seconds / total seconds) plus absolute
durations. `assess_quality()` (`app/audio/quality.py`) turns that into the
three-way grade:

| condition                                                            | grade          |
|---------------------------------------------------------------------|----------------|
| `speech_ratio >= 0.5`                                                | `good`         |
| `0.15 <= speech_ratio < 0.5`                                         | `degraded`     |
| `speech_ratio < 0.15`, **or** `< 0.5 s` of speech, **or** clip `< 0.5 s` | `insufficient` |

We gate on **speech ratio only** — deliberately not SNR, clipping, or level.
The model tolerates a low SNR (it was pre-trained for it) far better than it
tolerates *near-silence*, where it regresses to a confident-looking but
meaningless answer. "Is there actually a voice here?" is the question that
matters.

An `insufficient` grade **skips the model entirely** — the handler returns
`unknown` / `unknown` at `0.0` confidence without ever calling
`predict()` (`app/api/routes.py`, `_run_pipeline`, lines 124–125). Two
reasons: (1) it saves the forward-pass compute on audio that cannot produce
a useful answer, and (2) more importantly, running the model on silence
produces a *confident-looking* wrong answer, which is worse than an honest
"unknown" for a downstream consumer that might act on it.

**The thresholds (`0.15`, `0.5`, `0.5 s`) are a hand-picked starting point,
not tuned against human-labeled clips.** `tests/test_quality.py` pins the
exact boundary behavior (27 tests) so that a future retune is a deliberate,
visible change rather than an accident.

### Why the model loads once at startup

`app/main.py` loads the `AttributeInferencer` in the FastAPI **lifespan**
startup handler and stashes it on `app.state.inferencer`; every request
reuses that one instance. `predict()` keeps no per-request state, so this is
safe. Loading per-request would put a multi-second model construction on the
critical path of every call.

**Latency implication, measured on this machine (CPU, macOS, `sample.wav`, a
2.9 s clip):**

| phase                                  | time      |
|----------------------------------------|-----------|
| cold model load (startup, one-time)    | ~6.3 s    |
| first `/analyze` after boot (warm-up)  | ~2.2 s end-to-end |
| steady-state `/analyze`                | **~200–300 ms** end-to-end (~180–270 ms in the forward pass) |

So the cold-start cost is paid once, at boot, by the process — not by a
caller. `GET /health` returning `model_loaded: true` is the signal that the
warm path is available; an orchestrator should not route traffic until then.
(The Docker image goes further and *bakes* the weights in at build time so
the first boot does not even pay the download — see the Dockerfile comment
for that tradeoff.)

### Reliability guarantees (`app/api/routes.py`)

- **Never a 500.** The whole handler is wrapped in a broad `try/except`
  (line 267). Any unexpected error logs at `ERROR` with
  `outcome: "error_fallback"` and returns a 200 with the all-`unknown` /
  `insufficient` body. This service is a soft signal; a bug in it must not
  break a live call.
- **Never hangs.** The decode → VAD → inference core runs in a worker
  thread under a hard 3 s wall-clock budget
  (`_PROCESSING_BUDGET_S`, line 82). On overrun the request returns the
  safe body immediately; the orphaned thread finishes on its own and is
  discarded (the pipeline holds no shared mutable state).
- **One structured log line per request** (`app/logging_config.py`,
  JSON on stdout): `contact_id`, `audio_quality`, `gender_prediction`,
  `age_prediction`, `processing_ms`, `inference_ms`, `outcome`. `INFO` for
  a clean `good` read, `WARNING` for `degraded`/`insufficient`/decode
  failure/timeout, `ERROR` for the exception fallback.

## 5. Privacy

Call recordings routinely contain PII and are often subject to
recording-consent constraints. This service is built so the audio never
lands anywhere durable and never reaches the logs.

- **Audio is processed entirely in memory. It is never written to disk and
  no `tempfile` is used.**
  - The upload bytes are read into a local variable in the handler
    (`app/api/routes.py:174`, `raw_bytes = await audio.read()`) and passed
    by value down the pipeline.
  - Decoding and resampling happen by streaming those bytes into `ffmpeg`
    over `stdin` and reading PCM back over `stdout` — no scratch file.
    See `app/audio/ingest.py`, `normalize_audio()` (`subprocess.run(cmd,
    input=raw_bytes, stdout=PIPE, stderr=PIPE)`, lines ~140–146; the
    `pipe:0` / `pipe:1` ffmpeg arguments at lines 126 and 136).
  - The module docstrings state this contract explicitly:
    `app/audio/ingest.py:11-15` and `app/api/routes.py:10-13`.
- **No audio, and no raw PII, is ever logged. Only `contact_id` plus
  prediction metadata.**
  - The JSON formatter promotes a fixed allowlist of fields
    (`_PROMOTED_FIELDS` in `app/logging_config.py:28-45`); anything else
    passed via `extra=` is dropped. The upload filename, byte content, and
    any transcript are not on that list and are never passed in.
  - The only request identifier is the uuid4 the service generates itself
    (`app/api/routes.py:163`). A caller-supplied `contact_id` form field is
    accepted for compatibility but ignored, so a caller cannot inject an
    identifier into the logs.
  - The exception path logs `exc_info` (traceback) and the exception *type*
    only — never the request body (`app/api/routes.py:271-283`).
  - This is covered by a test:
    `tests/test_reliability.py::test_analyze_emits_one_structured_json_line_without_pii`
    uploads a file named `caller-jane-doe-ssn-123.wav` and asserts that
    none of `caller-jane`, `jane-doe`, `ssn`, `.wav`, or `filename` appears
    anywhere in the emitted log line.

## 6. Known limitations

- **Untested / likely-degraded conditions.** Accuracy on strong accents,
  heavy background noise (trucks, warehouses, forklifts), very short
  utterances, and telephony-compressed audio (8 kHz G.711 / other narrowband
  codecs) has **not** been measured. The model's robust pre-training is a
  reason to *expect* it holds up better than a studio-trained model, not
  evidence that it does. Treat all of these as unknown-and-probably-worse
  until there is an eval set that covers them.
- **The eval is not statistically meaningful.** As shipped, the manifest has
  **N = 1** clip (`sample.wav`). Even filled out to 10–20 clips it is an
  anecdote — the confidence interval on an accuracy from ~15 samples is
  roughly ±25 points. The harness prints a loud warning to this effect.
  There is also no class-balance check: a manifest of 12 men + 2 women would
  let an "always male" model score 86% and teach nothing.
- **On the one bundled clip the model is wrong.** `sample.wav` is the
  LDC93S1 TIMIT sentence (a ~46-year-old male speaker, 16 kHz, 2.9 s).
  `python eval/run_eval.py` on it returns `gender: "unknown"` (the model's
  internal "child" class wins) and `age_bracket: "18-30"`. It is a concrete
  example of the model struggling on short, telephone-ish audio — exactly
  what a real eval set would quantify.
- **Age confidence is a heuristic, not a calibrated probability.** See
  [§4](#4-architecture--design-decisions). Do not threshold on it as if it
  were `P(correct)`.
- **Gender is binary plus "unknown".** The model was trained on binary
  gender labels (its third class is "child", not a gender identity). Its
  outputs are limited to `male` / `female`, with `unknown` as the fallback
  for the child class and low-confidence cases. It does **not** represent
  nonbinary or other gender identities, and it should not be treated as
  doing so. This is a limitation of the training data, surfaced honestly
  rather than papered over.
- **No real telephony audio was tested.** Everything exercised so far is
  16 kHz+ source audio. No 8 kHz G.711 (µ-law / a-law) call capture has been
  run through the service end to end. The ingestion layer has code paths for
  it (`normalize_audio(input_format="mulaw"|"alaw")`), but they are
  untested against real SIP-trunk audio. Given that the use case *is* a call
  center, this is the most important gap to close next.
- **Quality-gate thresholds are uncalibrated** — a hand-picked starting
  point, not tuned on clips labeled `good` / `degraded` / `insufficient`.

## 7. Scaling to 1000 concurrent calls

The current process is one uvicorn worker doing synchronous CPU inference,
~200–300 ms per call. That caps a single process at a handful of
calls/second. To reach ~1000 concurrent calls (elaborated in the video):

- **Batch inference requests.** Collect clips arriving within a short window
  and run them as one padded batch through the model — throughput per
  forward pass scales far better than per-request calls, especially on a
  GPU.
- **Move inference to a dedicated GPU inference server.** Run the model
  behind Triton or TorchServe on GPU nodes instead of per-process CPU
  inference. The FastAPI service becomes a thin client that does ingest +
  VAD + quality gating and forwards the waveform to the inference tier.
- **Horizontal autoscaling behind a load balancer.** Stateless API
  replicas, `GET /health` (`model_loaded`) as the readiness gate,
  autoscaled on queue depth / latency. The GPU inference tier scales
  independently.
- **Quantize the model** (int8 / fp16) to cut per-inference latency and
  memory, allowing more concurrent streams per GPU.

## 8. Bonus tasks

| Task                                  | Status | Notes                                                                                     |
|---------------------------------------|--------|-----------------------------------------------------------------------------------------|
| Dockerization + docker-compose        | ✅ done | Weights baked at build time, named volume for cache, `/health`-gated healthcheck.        |
| Structured JSON logging / observability | ✅ done | One line per request, PII-safe, level reflects outcome. `app/logging_config.py`.        |
| Reliability hardening                  | ✅ done | Never-500 safety net + 3 s timeout guard, both tested (`tests/test_reliability.py`).     |
| Offline eval harness                   | ⚠️ partial | `eval/run_eval.py` is complete and honest (accuracy + abstain rate + calibration table), but the labeled dataset is a stub of N = 1. Mozilla Common Voice — the natural source — is now a gated HF dataset (account + token + licence + multi-GB download), so it was deliberately kept out of the zero-setup path; `--help` prints the ~15 lines to wire it in. |
| Streaming inference (`WS /ws/analyze`) | ✅ done | `app/api/websocket.py`. Client streams raw 16 kHz mono float32 PCM chunks; the server buffers them into non-overlapping 2 s windows and runs each through the **same** `normalize_audio → run_vad → assess_quality → predict()` pipeline as `/analyze` (thin orchestration, no duplicated logic). Emits an `/analyze`-shaped JSON message per window plus `window_index` and a running `best_so_far` (highest-gender-confidence-wins — see the `observe()` comment for why not averaging). Per-window worker-thread timeout, graceful client-disconnect handling, in-memory only (nothing to disk, no bytes logged), and session-duration / total-bytes / window-count guards so a client can't hold the socket open or exhaust memory. Test client: `scripts/test_websocket.py` streams `sample.wav` in ~0.5 s chunks in real time. See [§3](#ws-wsanalyze). |
| Quality-threshold calibration          | ❌ skipped | Needs a set of clips human-labeled `good`/`degraded`/`insufficient`, which does not exist yet. Thresholds are a documented starting point; boundaries are pinned by tests so a retune is deliberate. |
