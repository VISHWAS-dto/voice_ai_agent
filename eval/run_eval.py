"""Offline evaluation harness for the voice attribute inference model.

WHAT THIS IS
------------
A small, honest, offline check of ``AttributeInferencer`` (the wav2vec2
age/gender wrapper in ``app/inference/model.py``). It reads a manifest of
labeled clips, runs each one through the inferencer **directly** (not via
the HTTP API - we want to measure the model, not the web stack), and
prints:

  * gender accuracy   - % correct over clips where the model committed to
                        male/female, with the "unknown" (abstain) rate
                        reported separately so abstentions can't be
                        confused with either right or wrong answers;
  * age-bracket accuracy - same idea, exact-bracket match, abstain rate
                        reported separately;
  * a calibration table - predictions bucketed by the confidence score the
                        model reported (0.5-0.6, 0.6-0.7, ...), with the
                        actual hit-rate inside each bucket. If the model
                        says "0.9" on a bunch of clips and only gets 60% of
                        them right, its confidence is not trustworthy, and
                        this table is where you'd see that.

BE HONEST ABOUT WHAT THIS ISN'T
-------------------------------
  * It is NOT statistically meaningful. As shipped, the manifest has a
    single clip. Even a filled-out 10-20 clip set is an anecdote, not an
    evaluation - the confidence intervals on an accuracy computed from 15
    samples are enormous (roughly +/- 25 points at n=15). Every number
    below should be read as "here is what happened on these specific
    clips", never as "the model is X% accurate".
  * It has NO class balance guarantees. If your manifest is 12 men and 2
    women, a model that always guesses "male" scores 86% and learns
    nothing.
  * The bundled label is unverified (see manifest.csv). Age brackets in
    particular are guesses unless you supply verified ones.
  * Mozilla Common Voice - the natural real source of age/gender-labeled
    speech - is a GATED Hugging Face dataset now: it needs an account, a
    token, an accepted licence, the ``datasets`` package, and a multi-GB
    download. That was deliberately NOT wired in as a default so this
    script runs with zero setup. If you have a token and want a real run,
    see ``_commonvoice_hint()`` at the bottom of this file for the ~15
    lines that would extend the manifest from Common Voice.

USAGE
-----
    python eval/run_eval.py
    python eval/run_eval.py --manifest eval/datasets/manifest.csv
    python eval/run_eval.py --json eval/reports/run.json   # also dump raw results

Requires the same environment as the service (ffmpeg on PATH, model weights
cached in ``.model_cache/``). A clip that fails to decode or infer is
logged and skipped - one bad file never aborts the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "eval" / "datasets" / "manifest.csv"

# The four real brackets the API can emit (plus "unknown", handled as abstain).
_AGE_BRACKETS = {"18-30", "31-45", "46-60", "60+"}
_GENDERS = {"male", "female"}

# Confidence buckets for the calibration table. Predictions with a
# confidence below 0.5 land in the "<0.5" catch-all row.
_CALIB_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


@dataclass
class ClipLabel:
    """One manifest row: a clip path plus its ground-truth labels."""

    path: Path
    gender: str | None          # "male" | "female" | None (unknown/excluded)
    age_bracket: str | None     # one of _AGE_BRACKETS, or None
    verified: bool
    notes: str


@dataclass
class ClipResult:
    """What the model said about one clip, alongside the truth."""

    label: ClipLabel
    ok: bool                    # did decode + inference succeed at all?
    error: str | None = None

    gender_pred: str | None = None
    gender_conf: float | None = None
    age_pred: str | None = None
    age_conf: float | None = None
    inference_ms: float | None = None


@dataclass
class Tally:
    """Running counts for one attribute (gender OR age)."""

    labeled: int = 0            # rows that had a usable ground-truth label
    scored_correct: int = 0     # model committed AND matched truth
    scored_wrong: int = 0       # model committed AND missed
    abstained: int = 0          # model said "unknown" on a labeled row

    @property
    def committed(self) -> int:
        return self.scored_correct + self.scored_wrong

    @property
    def accuracy_excl_unknown(self) -> float | None:
        """Accuracy over rows where the model actually committed."""
        if self.committed == 0:
            return None
        return self.scored_correct / self.committed

    @property
    def abstain_rate(self) -> float | None:
        if self.labeled == 0:
            return None
        return self.abstained / self.labeled


@dataclass
class CalibBucket:
    """One row of the calibration table."""

    lo: float
    hi: float
    n: int = 0
    correct: int = 0

    @property
    def hit_rate(self) -> float | None:
        return self.correct / self.n if self.n else None


# --------------------------------------------------------------------------
# manifest loading
# --------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> list[ClipLabel]:
    """Parse the CSV manifest into ClipLabel rows.

    Tolerant on purpose: comment/blank lines are skipped, a missing file is
    reported but does not abort the load, and malformed label cells degrade
    to "excluded from that metric" rather than raising.
    """
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    labels: list[ClipLabel] = []
    with manifest_path.open(newline="") as fh:
        # Strip comment lines before handing to csv so '#' comments work
        # even though csv has no native comment support.
        rows = (ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#"))
        reader = csv.reader(rows)
        for lineno, row in enumerate(reader, start=1):
            if len(row) < 4:
                print(f"  [manifest] skipping malformed row {lineno}: {row!r}")
                continue
            raw_path, raw_gender, raw_age, raw_verified = (c.strip() for c in row[:4])
            notes = row[4].strip() if len(row) > 4 else ""

            # Skip a header row if the manifest carries one uncommented
            # (the shipped manifest comments it out, but a hand-edited one
            # might not).
            if raw_path.lower() == "path" and raw_gender.lower() == "gender":
                continue

            clip_path = Path(raw_path)
            if not clip_path.is_absolute():
                clip_path = REPO_ROOT / clip_path

            gender = raw_gender.lower() if raw_gender.lower() in _GENDERS else None
            age = raw_age if raw_age in _AGE_BRACKETS else None
            verified = raw_verified.lower() in {"yes", "y", "true", "1"}

            labels.append(
                ClipLabel(
                    path=clip_path,
                    gender=gender,
                    age_bracket=age,
                    verified=verified,
                    notes=notes,
                )
            )
    return labels


# --------------------------------------------------------------------------
# running the model
# --------------------------------------------------------------------------


def run_clips(labels: list[ClipLabel]) -> list[ClipResult]:
    """Decode + infer every clip. Failures are captured per-clip, never raised."""
    # Imported here (not at module top) so ``--help`` and a bad manifest
    # path don't pay the ~1.3 GB model load.
    from app.audio.ingest import AudioDecodeError, normalize_audio
    from app.inference.model import AttributeInferencer

    print(f"loading AttributeInferencer (one-time, ~1.3 GB from .model_cache/) ...")
    t0 = time.perf_counter()
    try:
        inferencer = AttributeInferencer()
    except RuntimeError as exc:
        raise SystemExit(f"could not load the model: {exc}")
    print(f"  model ready in {time.perf_counter() - t0:.1f}s\n")

    results: list[ClipResult] = []
    for i, label in enumerate(labels, start=1):
        tag = f"[{i}/{len(labels)}] {label.path.name}"
        if not label.path.is_file():
            print(f"  {tag}: MISSING FILE - skipped")
            results.append(ClipResult(label=label, ok=False, error="file not found"))
            continue

        try:
            waveform = normalize_audio(label.path.read_bytes())
        except AudioDecodeError as exc:
            print(f"  {tag}: decode failed ({exc}) - skipped")
            results.append(ClipResult(label=label, ok=False, error=f"decode: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            print(f"  {tag}: unexpected read/decode error ({exc!r}) - skipped")
            results.append(ClipResult(label=label, ok=False, error=f"read: {exc!r}"))
            continue

        try:
            pred = inferencer.predict(waveform, 16_000)
        except Exception as exc:  # noqa: BLE001 - predict() shouldn't raise, but be safe
            print(f"  {tag}: inference error ({exc!r}) - skipped")
            results.append(ClipResult(label=label, ok=False, error=f"predict: {exc!r}"))
            continue

        res = ClipResult(
            label=label,
            ok=True,
            gender_pred=pred["gender_prediction"],
            gender_conf=float(pred["gender_confidence"]),
            age_pred=pred["age_bracket"],
            age_conf=float(pred["age_confidence"]),
            inference_ms=float(pred.get("inference_ms", 0.0)),
        )
        results.append(res)
        print(
            f"  {tag}: gender={res.gender_pred}({res.gender_conf:.2f}) "
            f"age={res.age_pred}({res.age_conf:.2f}) "
            f"truth=({label.gender or '?'}/{label.age_bracket or '?'})"
        )
    print()
    return results


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def tally_attribute(
    results: list[ClipResult], attr: str, *, verified_only: bool = False
) -> Tally:
    """Build a Tally for 'gender' or 'age' over the successful results.

    A row counts toward ``labeled`` only if it has a ground-truth label for
    this attribute. "unknown" from the model is an abstention, tracked
    separately from right/wrong.
    """
    t = Tally()
    for r in results:
        if not r.ok:
            continue
        if verified_only and not r.label.verified:
            continue

        if attr == "gender":
            truth, pred = r.label.gender, r.gender_pred
        else:
            truth, pred = r.label.age_bracket, r.age_pred

        if truth is None:
            continue
        t.labeled += 1

        if pred is None or pred == "unknown":
            t.abstained += 1
        elif pred == truth:
            t.scored_correct += 1
        else:
            t.scored_wrong += 1
    return t


def calibration_table(
    results: list[ClipResult], attr: str
) -> list[CalibBucket]:
    """Bucket committed predictions by reported confidence; count hits.

    Only rows that (a) succeeded, (b) have a ground-truth label, and (c)
    got a committed (non-"unknown") prediction contribute. An "unknown"
    prediction has no meaningful confidence to bucket.
    """
    buckets = [
        CalibBucket(lo=_CALIB_EDGES[i], hi=_CALIB_EDGES[i + 1])
        for i in range(len(_CALIB_EDGES) - 1)
    ]
    below = CalibBucket(lo=0.0, hi=0.5)  # catch-all for conf < 0.5

    for r in results:
        if not r.ok:
            continue
        if attr == "gender":
            truth, pred, conf = r.label.gender, r.gender_pred, r.gender_conf
        else:
            truth, pred, conf = r.label.age_bracket, r.age_pred, r.age_conf

        if truth is None or pred is None or pred == "unknown" or conf is None:
            continue

        hit = int(pred == truth)
        if conf < 0.5:
            below.n += 1
            below.correct += hit
            continue
        for b in buckets:
            if b.lo <= conf < b.hi:
                b.n += 1
                b.correct += hit
                break

    return [below, *buckets]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def print_report(labels: list[ClipLabel], results: list[ClipResult]) -> None:
    n_total = len(labels)
    n_ok = sum(r.ok for r in results)
    n_failed = n_total - n_ok
    n_verified = sum(r.ok and r.label.verified for r in results)

    print("=" * 68)
    print("EVAL REPORT")
    print("=" * 68)
    print(f"clips in manifest      : {n_total}")
    print(f"clips scored           : {n_ok}   (failed/skipped: {n_failed})")
    print(f"  of which verified     : {n_verified}")
    print()
    print("!! Tiny sample. Not statistically significant. These numbers")
    print("!! describe THESE clips only - do not quote them as the model's")
    print("!! accuracy. See the module docstring in eval/run_eval.py.")
    print()

    for scope_label, verified_only in (("ALL scored clips", False),
                                       ("VERIFIED-label clips only", True)):
        g = tally_attribute(results, "gender", verified_only=verified_only)
        a = tally_attribute(results, "age", verified_only=verified_only)
        print("-" * 68)
        print(f"{scope_label}")
        print("-" * 68)
        print(f"  GENDER   labeled={g.labeled}  committed={g.committed}  "
              f"abstained(unknown)={g.abstained}")
        print(f"           accuracy (excl. unknown) : {_pct(g.accuracy_excl_unknown)}"
              f"   [{g.scored_correct}/{g.committed}]")
        print(f"           abstain rate             : {_pct(g.abstain_rate)}"
              f"   [{g.abstained}/{g.labeled}]")
        print(f"  AGE      labeled={a.labeled}  committed={a.committed}  "
              f"abstained(unknown)={a.abstained}")
        print(f"           accuracy (excl. unknown) : {_pct(a.accuracy_excl_unknown)}"
              f"   [{a.scored_correct}/{a.committed}]")
        print(f"           abstain rate             : {_pct(a.abstain_rate)}"
              f"   [{a.abstained}/{a.labeled}]")
        print()

    # Calibration - over ALL scored clips (verified + not), since we need
    # every data point we can get and this table is diagnostic, not a score.
    for attr, human in (("gender", "GENDER"), ("age", "AGE BRACKET")):
        print("-" * 68)
        print(f"CALIBRATION - {human}  (reported-confidence bucket vs. actual hit rate)")
        print("-" * 68)
        table = calibration_table(results, attr)
        print(f"  {'conf bucket':<14}{'n':>5}{'correct':>9}{'actual acc':>13}")
        any_rows = False
        for b in table:
            if b.n == 0:
                continue
            any_rows = True
            name = f"<0.5" if b.hi == 0.5 else f"{b.lo:.1f}-{b.hi:.1f}".replace("1.0001", "1.0")
            print(f"  {name:<14}{b.n:>5}{b.correct:>9}{_pct(b.hit_rate):>13}")
        if not any_rows:
            print("  (no committed predictions to bucket)")
        print()

    print("-" * 68)
    print("PER-CLIP DETAIL")
    print("-" * 68)
    for r in results:
        if not r.ok:
            print(f"  {r.label.path.name:<28} FAILED: {r.error}")
            continue
        g_ok = _mark(r.gender_pred, r.label.gender)
        a_ok = _mark(r.age_pred, r.label.age_bracket)
        vflag = "" if r.label.verified else "  (label unverified)"
        print(f"  {r.label.path.name:<28} "
              f"gender {r.gender_pred:>7}({r.gender_conf:.2f}){g_ok}  "
              f"age {str(r.age_pred):>7}({r.age_conf:.2f}){a_ok}{vflag}")
    print("=" * 68)


def _mark(pred: str | None, truth: str | None) -> str:
    if truth is None:
        return " -"
    if pred is None or pred == "unknown":
        return " ?"   # abstained
    return " OK" if pred == truth else " X"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _results_to_jsonable(labels, results) -> dict:
    return {
        "manifest_clips": len(labels),
        "scored": sum(r.ok for r in results),
        "clips": [
            {
                "path": str(r.label.path),
                "truth": {"gender": r.label.gender, "age_bracket": r.label.age_bracket,
                          "verified": r.label.verified},
                "ok": r.ok,
                "error": r.error,
                "pred": None if not r.ok else {
                    "gender": r.gender_pred, "gender_conf": r.gender_conf,
                    "age_bracket": r.age_pred, "age_conf": r.age_conf,
                    "inference_ms": r.inference_ms,
                },
            }
            for r in results
        ],
    }


def _commonvoice_hint() -> str:
    return (
        "To run against Mozilla Common Voice instead of the bundled stub set:\n"
        "  1. `pip install datasets soundfile`\n"
        "  2. Get a Hugging Face token and accept the dataset terms at\n"
        "     https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0\n"
        "  3. `huggingface-cli login` (or set HF_TOKEN)\n"
        "  4. Roughly:\n"
        "       from datasets import load_dataset\n"
        "       ds = load_dataset('mozilla-foundation/common_voice_17_0', 'en',\n"
        "                         split='validated', streaming=True)\n"
        "       # keep only rows with non-empty `age` and `gender`, take ~20,\n"
        "       # write each `audio` array to eval/datasets/*.wav, and append\n"
        "       # rows to manifest.csv mapping CV's age strings\n"
        "       #   ('twenties'->'18-30', 'thirties'/'fourties'->'31-45',\n"
        "       #    'fifties'->'46-60', 'sixties'+/'seventies'+->'60+')\n"
        "     then re-run this script.\n"
        "  NOTE: CV age/gender is SELF-REPORTED and skews young + male; even a\n"
        "  20-clip pull is still just a spot check, not a benchmark."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline eval for the voice attribute inference model.",
        epilog=_commonvoice_hint(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"CSV manifest of labeled clips (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="also write raw per-clip results to this path as JSON",
    )
    args = parser.parse_args(argv)

    print(f"manifest: {args.manifest}")
    labels = load_manifest(args.manifest)
    if not labels:
        print("\nmanifest has no usable rows. Add labeled clips - see the "
              "comments in eval/datasets/manifest.csv and, for Common Voice, "
              "`python eval/run_eval.py --help`.")
        return 1
    print(f"loaded {len(labels)} clip(s)\n")

    results = run_clips(labels)
    print_report(labels, results)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_results_to_jsonable(labels, results), indent=2))
        print(f"\nraw results written to {args.json}")

    # Exit non-zero only if literally nothing scored - a low accuracy on a
    # tiny set is not a CI failure condition, and pretending otherwise
    # would be the dishonest move this harness is trying to avoid.
    return 0 if any(r.ok for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
