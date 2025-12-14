"""
Learning path in code:
1) Load pretrained Emformer-RNN-T (TorchAudio bundle)
2) Build real streaming loop (mic -> decode)
3) Measure first-token latency + rolling stats
4) Tune knobs: beam_width, commit_stability, segment hop
5) Stress test: optional CPU hog thread; verify you drop rather than buffer

Run:
  pip install torch torchaudio sounddevice numpy
  python learn_streaming_rnnt.py

Notes:
- macOS isn't real-time. So we refuse to build infinite latency:
  if audio_q is full we drop mic frames.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from collections import deque


# -------------------------
# Metrics
# -------------------------

@dataclass
class LatencyStats:
    """Tracks first-token latency (audio timestamp -> first non-empty text) and rolling decode times."""
    first_token_lat_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    step_compute_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    dropped_audio_chunks: int = 0
    steps: int = 0

    def add_first_token(self, ms: float) -> None:
        self.first_token_lat_ms.append(ms)

    def add_step_compute(self, ms: float) -> None:
        self.step_compute_ms.append(ms)
        self.steps += 1

    def summary(self) -> str:
        def pct(xs, p):
            if not xs:
                return float("nan")
            arr = np.asarray(xs, dtype=np.float64)
            return float(np.percentile(arr, p))

        f = list(self.first_token_lat_ms)
        c = list(self.step_compute_ms)
        return (
            f"steps={self.steps} dropped_audio={self.dropped_audio_chunks} | "
            f"first_token_ms p50={pct(f,50):.1f} p95={pct(f,95):.1f} n={len(f)} | "
            f"compute_ms p50={pct(c,50):.1f} p95={pct(c,95):.1f} n={len(c)}"
        )


# -------------------------
# Utility: simple commit logic
# -------------------------

@dataclass
class Committer:
    """
    Avoids flicker by committing text only when stable for N steps.
    This is intentionally simple: it commits full text once stable.
    """
    stability_steps: int = 3
    last_text: str = ""
    stable_count: int = 0
    committed: str = ""

    def update(self, text: str) -> Tuple[str, str]:
        """Returns (committed, tail_uncommitted)."""
        if text == self.last_text:
            self.stable_count += 1
        else:
            self.last_text = text
            self.stable_count = 0

        if self.stable_count >= self.stability_steps:
            self.committed = text

        tail = text[len(self.committed):] if text.startswith(self.committed) else text
        return self.committed, tail


# -------------------------
# Optional stress test
# -------------------------

def cpu_hog(stop_evt: threading.Event, intensity: float = 0.75) -> None:
    """
    Dumb CPU load generator. intensity in (0,1): fraction of time spent busy.
    The point is not elegance. The point is to see your latency degrade and whether you fall behind.
    """
    busy = max(0.0, min(0.99, intensity))
    period = 0.02  # 20 ms
    busy_t = period * busy
    idle_t = period - busy_t

    while not stop_evt.is_set():
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < busy_t:
            # pointless arithmetic
            _ = 1234567 * 7654321
        time.sleep(idle_t)


# -------------------------
# Main streaming prototype
# -------------------------

def run_streaming(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    device = torch.device("cpu")  # keep prototype simple

    # 1) Load pretrained model bundle
    bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
    sample_rate = bundle.sample_rate
    hop_length = bundle.hop_length
    seg_len_frames = bundle.segment_length
    right_ctx_frames = bundle.right_context_length

    seg_len_samples = seg_len_frames * hop_length
    window_samples = (seg_len_frames + right_ctx_frames) * hop_length

    streaming_fe = bundle.get_streaming_feature_extractor()
    decoder = bundle.get_decoder()
    token_processor = bundle.get_token_processor()

    # 2) Audio capture -> queue
    audio_q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(maxsize=args.audio_queue_max)

    stats = LatencyStats()
    committer = Committer(stability_steps=args.commit_stability)

    # Resolve input device (allow numeric or name)
    sd_device = args.device
    if isinstance(sd_device, str) and sd_device.isdigit():
        sd_device = int(sd_device)
    device_sr = sample_rate
    if sd_device is not None:
        info = sd.query_devices(sd_device, "input")
        device_sr = int(info["default_samplerate"])
        print(f"Using input device: {info['name']} (id={sd_device}, default_sr={info['default_samplerate']})")
    else:
        # If no device specified, infer default input samplerate
        try:
            info = sd.query_devices(None, "input")
            device_sr = int(info["default_samplerate"])
            print(f"Default input device: {info['name']} (sr={info['default_samplerate']})")
        except Exception:
            device_sr = sample_rate

    def audio_callback(indata, frames, t, status):
        # Timestamp: "when this chunk entered the callback"
        ts = time.perf_counter()
        mono = indata[:, 0].astype(np.float32, copy=False)
        try:
            audio_q.put_nowait((mono, ts))
        except queue.Full:
            stats.dropped_audio_chunks += 1
            # Drop. Never buffer forever.

    # Capture at device native rate to avoid driver-side resampler; resample in main loop if needed.
    stream = sd.InputStream(
        samplerate=device_sr,
        channels=1,
        dtype="float32",
        blocksize=0,
        device=sd_device,
        callback=audio_callback,
    )

    # Ring buffer holds exactly one inference window (segment + right context)
    ring = np.zeros(window_samples, dtype=np.float32)
    ring_fill = 0
    earliest_ts_in_window: Optional[float] = None

    # Streaming decode state
    state = None
    hypotheses = None  # list of hypotheses to carry across streaming steps

    # First token latency measurement (per “utterance-ish”)
    # For a prototype, we reset the "first token timer" when we see silence-ish behavior.
    # (No VAD here; we just reset when committed text stops changing for a while.)
    last_emit_time = time.perf_counter()
    waiting_for_first_token = True
    window_start_ts: Optional[float] = None

    # 5) Optional stress thread
    hog_stop = threading.Event()
    hog_thread = None
    if args.stress_cpu:
        hog_thread = threading.Thread(target=cpu_hog, args=(hog_stop, args.stress_intensity), daemon=True)
        hog_thread.start()

    print("\nListening... Ctrl+C to stop.\n")
    print(f"bundle=EMFORMER_RNNT_BASE_LIBRISPEECH sr={sample_rate} "
          f"segment={seg_len_frames} frames ({seg_len_samples/sample_rate:.3f}s) "
          f"right_ctx={right_ctx_frames} frames "
          f"beam={args.beam_width} commit_stability={args.commit_stability}\n")

    next_report = time.perf_counter() + args.report_every

    with stream, torch.inference_mode():
        try:
            while True:
                # Fill window
                while ring_fill < window_samples:
                    chunk, ts = audio_q.get()
                    if earliest_ts_in_window is None:
                        earliest_ts_in_window = ts
                    if window_start_ts is None:
                        window_start_ts = ts  # timestamp when we began assembling a new window

                    # Resample to model sample_rate if needed
                    if device_sr != sample_rate:
                        chunk_tensor = torch.from_numpy(chunk)
                        chunk_tensor = torchaudio.functional.resample(chunk_tensor, device_sr, sample_rate)
                        chunk = chunk_tensor.numpy()

                    take = min(len(chunk), window_samples - ring_fill)
                    ring[ring_fill:ring_fill + take] = chunk[:take]
                    ring_fill += take

                # 3) Streaming inference step + compute timing
                t0 = time.perf_counter()
                window_rms = float(np.sqrt(np.mean(ring ** 2)))

                waveform = torch.from_numpy(ring.copy()).to(device)
                features, length = streaming_fe(waveform)
                hyps, state = decoder.infer(
                    features, length,
                    beam_width=args.beam_width,
                    state=state,
                    hypothesis=hypotheses,
                )
                hypotheses = hyps  # keep full beam for next step
                best_hypothesis = hyps[0]
                text = token_processor(best_hypothesis[0]).strip()

                t1 = time.perf_counter()
                stats.add_step_compute((t1 - t0) * 1000.0)

                if args.debug_log and (stats.steps % max(1, args.debug_every) == 0):
                    print(f"\n[debug] step={stats.steps} rms={window_rms:.4f} tokens={len(best_hypothesis[0])} text='{text}'")

                # 4) Latency: first token after a "reset"
                if waiting_for_first_token and text:
                    # Use window_start_ts as "audio arrival time" for this decode segment
                    if window_start_ts is not None:
                        stats.add_first_token((t1 - window_start_ts) * 1000.0)
                    waiting_for_first_token = False

                # Commit logic
                committed, tail = committer.update(text)
                if text:
                    last_emit_time = t1

                # Reset "utterance-ish" if no new committed text for a while
                # (Prototype heuristic; in a real system you'd use VAD/turn-taking.)
                if (t1 - last_emit_time) > args.reset_after_silence:
                    waiting_for_first_token = True
                    window_start_ts = None
                    # Also reset hypothesis if you want utterance segmentation:
                    if args.reset_hypothesis_on_silence:
                        hypotheses = None
                        state = None
                        committer = Committer(stability_steps=args.commit_stability)

                # Display
                if text:
                    display = committed + (f" [{tail}]" if tail else "")
                    print("\r" + display[-180:], end="", flush=True)

                # Advance window by segment hop (keep right-context overlap)
                ring[:-seg_len_samples] = ring[seg_len_samples:]
                ring_fill = window_samples - seg_len_samples
                earliest_ts_in_window = None  # will be set when we refill
                # Keep window_start_ts as timestamp of next step when we start refilling
                window_start_ts = None

                # Periodic report
                now = time.perf_counter()
                if now >= next_report:
                    print("\n" + stats.summary())
                    next_report = now + args.report_every

        except KeyboardInterrupt:
            print("\n\nStopped.")
            print(stats.summary())
        finally:
            hog_stop.set()
            if hog_thread is not None:
                hog_thread.join(timeout=0.2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="macOS streaming Emformer-RNNT learning path prototype")
    p.add_argument("--beam-width", type=int, default=4, help="RNNT beam width (1=greedy-ish, lower latency)")
    p.add_argument("--commit-stability", type=int, default=3, help="steps of identical partial before commit")
    p.add_argument("--audio-queue-max", type=int, default=200, help="max queued mic chunks before dropping")
    p.add_argument("--report-every", type=float, default=10.0, help="seconds between latency stats reports")
    p.add_argument("--reset-after-silence", type=float, default=1.2, help="seconds of no text before reset timer")
    p.add_argument("--reset-hypothesis-on-silence", action="store_true",
                   help="reset model hypothesis/state after silence (utterance-like segmentation)")
    p.add_argument("--stress-cpu", action="store_true", help="add CPU load to test robustness")
    p.add_argument("--stress-intensity", type=float, default=0.75, help="CPU hog intensity (0..0.99)")
    p.add_argument("--device", type=str, default=None, help="sounddevice input device (index or name)")
    p.add_argument("--debug-log", action="store_true", help="print debug info every N steps")
    p.add_argument("--debug-every", type=int, default=30, help="debug interval in steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_streaming(args)
