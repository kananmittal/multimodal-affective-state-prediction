"""
Local runner for the VER group-emotion pipeline.

The Music4D demo scripts in this folder were written against the Linux/CUDA
capture rig (hardcoded `device_map="cuda"`, live camera indices, an offline
PaliGemma snapshot at ../paligemma_offline). This runner reproduces the same
inference logic on a plain workstation: it auto-selects CUDA / MPS / CPU,
reads the bundled clips in Video/ instead of live cameras, and runs headless
by default so it works over SSH or in CI.

Usage:
    python3 run_local.py                       # 3 frames from each bundled clip
    python3 run_local.py --frames 5            # more samples per source
    python3 run_local.py --source Video/Audience.mp4
    python3 run_local.py --no-save             # print results only, write nothing
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

VALID_EMOTIONS = "joy, anger, fear, disgust, surprise, sadness, boredom, neutral"

# Same rule-based group-emotion prompt used by music4D-real-time-emo-qwen2-vl.py
PROMPT = (
    "Identify the general group emotion in this image following these specific rules: "
    "2. Distinguish 'sadness' (emotional pain, tears, downturned mouth, tense face) "
    "from 'boredom' (lack of stimulation, blank stare). If any sign of pain or distress "
    "is present, choose 'sadness'. "
    "3. Distinguish 'anger' (hostility, furrowed brows, tense jaw) from 'disgust' "
    "(revulsion, wrinkled nose, pulled-back lips). "
    "4. Use 'fear' only for reactions to a clear threat (defensive posture, shrinking "
    "back, wide eyes with tension). "
    "5. Use 'surprise' ONLY if the expression is *brief and shock-like*: wide eyes + "
    "open mouth + no signs of sustained pain, fear, anger, or disgust. "
    "   - If the same cues could indicate fear, sadness, or disgust, DO NOT choose 'surprise'. "
    "6. 'Neutral' is a LAST resort, only if there are no visible emotional cues at all. "
    f"Choose ONLY ONE or TWO emotion from the allowed list: {VALID_EMOTIONS}. "
    "Your answer MUST be in the format: 'emotions: [ em1, em2]'"
)

DEFAULT_SOURCES = {
    "DIRECTOR": "Video/director.mp4",
    "ORCHESTRA": "Video/musician.mp4",
    "AUDIENCE": "Video/Audience.mp4",
}


def pick_device():
    """CUDA where available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        # fp16 on MPS keeps the 2B model comfortably in unified memory.
        return "mps", torch.float16
    # fp16 matmuls are not implemented for most CPU kernels.
    return "cpu", torch.float32


def sample_frames(video_path, n):
    """Grab n frames spread evenly across the clip, as BGR ndarrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    # Avoid the very first/last frame, which are often black or a fade.
    idxs = [max(0, min(total - 1, int(total * (i + 1) / (n + 1)))) for i in range(n)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


def load_model(device, dtype, max_pixels):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    print(f"Loading {MODEL_ID} on {device} ({dtype})...", flush=True)
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    # Capping visual tokens is what keeps this real-time-ish; the original
    # script flagged min_pixels/max_pixels as critical on 8 GB VRAM.
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels * 28 * 28,
    )
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)
    return model, processor


def infer(model, processor, device, frame_bgr):
    """Run one group-emotion inference on a single BGR frame."""
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
    }]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()


def annotate(frame_bgr, source, label):
    out = frame_bgr.copy()
    h = out.shape[0]
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, source, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, h - 40), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, label[:70], (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 220, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3,
                    help="frames sampled per source (default 3)")
    ap.add_argument("--source", action="append",
                    help="video path; repeatable. Defaults to the three bundled clips.")
    ap.add_argument("--max-pixels", type=int, default=768,
                    help="visual-token budget in 28x28 patches (default 768)")
    ap.add_argument("--no-save", dest="no_save", action="store_true",
                    help="print results only; skip writing annotated frames and JSONL")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.source:
        sources = {Path(s).stem.upper(): s for s in args.source}
    else:
        sources = DEFAULT_SOURCES

    resolved = {}
    for name, rel in sources.items():
        p = (HERE / rel) if not Path(rel).is_absolute() else Path(rel)
        if not p.exists():
            print(f"skipping {name}: {p} not found", file=sys.stderr)
            continue
        resolved[name] = p
    if not resolved:
        print("no usable video sources", file=sys.stderr)
        return 1

    device, dtype = pick_device()
    model, processor = load_model(device, dtype, args.max_pixels)

    outdir = Path(args.outdir) if args.outdir else (
        HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not args.no_save:
        outdir.mkdir(parents=True, exist_ok=True)

    records = []
    for name, path in resolved.items():
        for idx, frame in sample_frames(path, args.frames):
            t0 = time.time()
            label = infer(model, processor, device, frame)
            dt = time.time() - t0
            print(f"[{name:9s}] frame {idx:4d}  {dt:5.1f}s  ->  {label}", flush=True)
            try:
                shown = str(path.relative_to(HERE))
            except ValueError:
                shown = str(path)
            rec = {"source": name, "video": shown,
                   "frame": idx, "seconds": round(dt, 2), "output": label}
            records.append(rec)
            if not args.no_save:
                img = annotate(frame, name, label)
                cv2.imwrite(str(outdir / f"{name}_{idx:05d}.jpg"), img)

    if not args.no_save:
        with open(outdir / "results.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nWrote {len(records)} annotated frames + results.jsonl to {outdir}")

    if records:
        avg = sum(r["seconds"] for r in records) / len(records)
        print(f"Mean inference time: {avg:.1f}s/frame on {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
