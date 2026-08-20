"""Review gallery: TDLA-v2 pages through paddleocr_v2 crops + OCR.

Samples interesting val/test pages (footnotes, two-column, header/footer)
from ``BDRC/TDLA-Training-Dataset-v2``, runs the same mask / column / footnote
path as the job, OCRs each image that would be sent to the model, and writes
a self-contained HTML page.

Usage (GPU box, after the vLLM engine is available)::

    /opt/pytorch/bin/python -m bec_orch.jobs.paddleocr.gallery_tdla \\
        --out-dir /tmp/paddleocr_v2_gallery --limit 100
"""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from bec_orch.jobs.paddleocr.config import PaddleOCRConfig
from bec_orch.jobs.paddleocr.layout_mask import (
    apply_header_footer_mask,
    removed_code,
    xywhn_to_xyxy,
)
from bec_orch.jobs.paddleocr.layout_split import (
    crop_body_regions,
    crop_labeled_regions,
    select_column_rects,
    text_area_rects,
)
from bec_orch.jobs.paddleocr.postprocess import normalize_text
from bec_orch.jobs.paddleocr.preprocess import bytes_to_rgb

REPO_ID = "BDRC/TDLA-Training-Dataset-v2"
CLS = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
COLORS = {
    "header": (220, 40, 40),
    "text-area": (30, 30, 30),
    "footnote": (20, 140, 50),
    "footer": (40, 80, 220),
}


def _load_hf_token() -> None:
    """Use WRITY_HF_TOKEN from the evaluation-benchmark env if HF_TOKEN is unset."""
    import os

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    env_path = Path("/home/eroux/BUDA/softs/ocr-evaluation-benchmark/env.sh")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("export WRITY_HF_TOKEN="):
            raw = line.split("=", 1)[1].strip().strip("'\"")
            if raw:
                os.environ["HF_TOKEN"] = raw
            return


def _download_labels(ds_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    _load_hf_token()
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        allow_patterns=["labels/val/**", "labels/test/**", "data.yaml", "val.txt", "test.txt"],
        local_dir=str(ds_dir),
    )
    return ds_dir


def _download_images(ds_dir: Path, rel_images: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    _load_hf_token()
    for rel in rel_images:
        dest = ds_dir / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(REPO_ID, rel, repo_type="dataset")
        if Path(path).resolve() != dest.resolve():
            shutil.copy2(path, dest)


def _parse_yolo(label_path: Path) -> list[dict]:
    boxes: list[dict] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except (TypeError, ValueError):
            continue
        lab = CLS.get(cid)
        if not lab or w <= 0 or h <= 0:
            continue
        boxes.append({"label": lab, "x": x, "y": y, "w": w, "h": h, "cls": cid})
    return boxes


def _is_two_column(boxes: list[dict]) -> bool:
    rects = text_area_rects(boxes, 1000, 1000, label="text-area")
    cols = select_column_rects(rects)
    return bool(cols) and len(cols) >= 2


def _tags(boxes: list[dict]) -> list[str]:
    labels = [b["label"] for b in boxes]
    tags: list[str] = []
    if labels.count("footnote"):
        tags.append(f"footnote×{labels.count('footnote')}")
    if _is_two_column(boxes):
        tags.append(f"two-col×{sum(1 for b in boxes if b['label']=='text-area')}")
    if "header" in labels:
        tags.append("header")
    if "footer" in labels:
        tags.append("footer")
    if not tags:
        tags.append("plain")
    return tags


def _score(boxes: list[dict]) -> tuple[int, str]:
    labels = [b["label"] for b in boxes]
    n_fn = labels.count("footnote")
    n_ta = labels.count("text-area")
    two = _is_two_column(boxes)
    has_h = "header" in labels
    has_f = "footer" in labels
    if n_fn and two:
        return 400 + n_fn * 10, "fn+cols"
    if n_fn:
        return 300 + n_fn * 10, "footnote"
    if two:
        return 200 + n_ta, "two-col"
    if has_h and has_f:
        return 120, "hf"
    if has_h:
        return 80, "h"
    if has_f:
        return 70, "f"
    if n_ta >= 3:
        return 50, "many-ta"
    return 10, "plain"


def _iter_label_pages(ds_dir: Path) -> list[dict]:
    pages: list[dict] = []
    for split in ("val", "test"):
        lab_dir = ds_dir / "labels" / split
        if not lab_dir.is_dir():
            continue
        for lab in sorted(lab_dir.glob("*.txt")):
            stem = lab.stem
            if "__aug" in stem:
                continue
            boxes = _parse_yolo(lab)
            if not boxes:
                continue
            rel = f"images/{split}/{stem}.jpg"
            score, bucket = _score(boxes)
            pages.append(
                {
                    "split": split,
                    "stem": stem,
                    "rel": rel,
                    "label_path": str(lab),
                    "boxes": boxes,
                    "score": score,
                    "bucket": bucket,
                    "tags": _tags(boxes),
                    "removed": removed_code(boxes),
                    "n_footnotes": sum(1 for b in boxes if b["label"] == "footnote"),
                    "two_col": _is_two_column(boxes),
                }
            )
    return pages


# Always keep these review pages when they appear in val/test (footer-in-footnote).
FORCE_STEMS = ("ldv1__W00EGS1016213__I1KG176__I1KG1760009",)


def _sample_pages(pages: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by[p["bucket"]].append(p)
    for bucket in by:
        rng.shuffle(by[bucket])
        by[bucket].sort(key=lambda x: -x["score"])

    picked: list[dict] = []
    seen: set[str] = set()
    by_stem = {p["stem"]: p for p in pages}
    for stem in FORCE_STEMS:
        p = by_stem.get(stem)
        if p is None or p["stem"] in seen:
            continue
        seen.add(p["stem"])
        picked.append(p)

    def take(bucket: str, n: int) -> None:
        for p in by.get(bucket, []):
            if len(picked) >= limit:
                return
            if p["stem"] in seen:
                continue
            seen.add(p["stem"])
            picked.append(p)
            n -= 1
            if n <= 0:
                return

    # Prefer two-column pages (the rare layout we most want to inspect),
    # then footnotes, then header/footer mix.
    take("fn+cols", 20)
    take("two-col", 45)
    take("footnote", 20)
    take("hf", 8)
    take("h", 4)
    take("f", 3)
    take("many-ta", 5)
    take("plain", 3)
    if len(picked) < limit:
        rest = [p for p in pages if p["stem"] not in seen]
        rest.sort(key=lambda x: (-int(x["two_col"]), -x["score"]))
        for p in rest:
            if len(picked) >= limit:
                break
            seen.add(p["stem"])
            picked.append(p)
    picked.sort(key=lambda x: (-x["score"], x["stem"]))
    return picked[:limit]


def _save_jpeg(img: Image.Image, path: Path, max_side: int = 1600) -> None:
    im = img.copy()
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    if im.mode != "RGB":
        im = im.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=88, optimize=True)


def _annotate(img: Image.Image, boxes: list[dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for b in boxes:
        lab = b.get("label") or ""
        xyxy = xywhn_to_xyxy(float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]), w, h)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=COLORS.get(lab, (0, 0, 0)), width=3)
    return out


def _v2_units(
    img: Image.Image, boxes: list[dict], cfg: PaddleOCRConfig
) -> tuple[list[dict], int, int, int]:
    """Build the list of images the model would see, in OCR order."""
    after, n_masked = apply_header_footer_mask(
        img,
        boxes,
        mask_labels=cfg.layout_mask_labels,
        protect_labels=cfg.layout_mask_protect_labels,
        pad_px=cfg.layout_mask_pad_px,
    )
    fn_crops = crop_labeled_regions(
        after,
        boxes,
        label=cfg.layout_footnote_label,
        margin_frac=cfg.layout_footnote_margin_frac,
        min_px=cfg.layout_footnote_margin_min_px,
        box_margin_frac=cfg.layout_footnote_box_margin_frac,
    )
    body_crops = crop_body_regions(
        after,
        boxes,
        label=cfg.layout_column_label,
        min_vert_overlap=cfg.layout_column_min_vert_overlap,
        max_horiz_overlap=cfg.layout_column_max_horiz_overlap,
        margin_frac=cfg.layout_column_margin_frac,
    )
    units: list[dict] = []
    n_columns = len(body_crops) if body_crops else 1
    if body_crops:
        # 2 crops = a column pair; 3+ = reading-order regions (XY-cut).
        kind = "column" if len(body_crops) == 2 else (
            "text-area" if len(body_crops) == 1 else "region"
        )
        for i, im in enumerate(body_crops):
            units.append({"kind": kind, "index": i, "image": im})
    else:
        units.append({"kind": "page", "index": 0, "image": after})
    for i, im in enumerate(fn_crops):
        units.append({"kind": "footnote", "index": i, "image": im})
    return units, n_masked, n_columns, len(fn_crops)


def _ocr_images(worker, images: list[Image.Image]) -> list[str]:
    if not images:
        return []
    prompts = [
        {"prompt": worker.prompt_text, "multi_modal_data": {"image": im}} for im in images
    ]
    sps = [
        worker._make_sampling(worker.cfg.temperature, n=1) for _ in prompts
    ]
    outputs = worker.llm.generate(prompts, sps, use_tqdm=False)
    texts = []
    for o in outputs:
        raw = o.outputs[0].text if o.outputs else ""
        texts.append(normalize_text(raw or ""))
    return texts


def _write_html(out_dir: Path, records: list[dict]) -> Path:
    n = len(records)
    n_fn = sum(1 for r in records if r["n_footnotes"])
    n_col = sum(1 for r in records if r["n_columns"] >= 2)
    cards = []
    for i, rec in enumerate(records):
        tag_html = "".join(
            f'<span class="tag {html.escape(t.split("×")[0])}">{html.escape(t)}</span>'
            for t in rec["tags"]
        )
        units_html = []
        order = 0
        for u in rec["units"]:
            is_footnote = u["kind"] == "footnote"
            if is_footnote:
                badge = f"<span class='order fn'>fn {u['index'] + 1}</span>"
            else:
                order += 1
                badge = f"<span class='order'>{order}</span>"
            units_html.append(
                "<figure class='unit'>"
                f"<figcaption>{html.escape(u['kind'])} {u['index']}</figcaption>"
                "<div class='imgwrap'>"
                f"{badge}"
                f"<img src='{html.escape(u['file'])}' alt='{html.escape(u['kind'])}'>"
                "</div>"
                f"<pre>{html.escape(u.get('text') or '(no OCR)')}</pre>"
                "</figure>"
            )
        filt = " ".join(
            [
                rec["bucket"],
                "has-fn" if rec["n_footnotes"] else "",
                "has-col" if rec["n_columns"] >= 2 else "",
                f"removed-{rec['removed']}",
            ]
        )
        cards.append(
            f"<article class='card' data-filter='{html.escape(filt)}' id='p{i}'>"
            f"<header><h2>{i + 1}. {html.escape(rec['stem'])}</h2>"
            f"<div class='meta'>{tag_html} "
            f"<span class='tag'>removed={html.escape(rec['removed'])}</span> "
            f"<span class='tag'>{rec['n_columns']} col</span> "
            f"<span class='tag'>{rec['n_units']} model image(s)</span> "
            f"<span class='tag'>{html.escape(rec['split'])}</span></div></header>"
            f"<div class='row'>"
            f"<figure class='orig'><figcaption>original + GT boxes</figcaption>"
            f"<img src='{html.escape(rec['original'])}' alt='original'></figure>"
            f"<div class='units'>{''.join(units_html)}</div>"
            f"</div></article>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>paddleocr_v2 gallery — {n} TDLA-v2 pages</title>
<style>
:root {{ font-family: "Noto Sans", "Noto Sans Tibetan", system-ui, sans-serif; }}
body {{ margin: 0; background: #f4f1ea; color: #1b1b1b; }}
nav {{ position: sticky; top: 0; z-index: 5; background: #2c2416; color: #fff;
      padding: 10px 16px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
nav h1 {{ font-size: 16px; margin: 0 12px 0 0; font-weight: 600; }}
nav button {{ border: 0; padding: 6px 10px; border-radius: 14px; cursor: pointer;
             background: #5c4a2e; color: #fff; }}
nav button.on {{ background: #e6c07b; color: #1b1b1b; }}
.card {{ background: #fff; margin: 16px; padding: 14px; border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
.card.hidden {{ display: none; }}
h2 {{ margin: 0 0 6px; font-size: 15px; font-weight: 600; word-break: break-all; }}
.meta {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }}
.tag {{ background: #eee; border-radius: 10px; padding: 2px 8px; font-size: 12px; }}
.tag.footnote {{ background: #d8f3d8; }}
.tag.two-col {{ background: #dce8f8; }}
.tag.header {{ background: #f8d4d4; }}
.tag.footer {{ background: #d4dcf8; }}
.row {{ display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }}
.orig {{ flex: 0 0 280px; max-width: 32vw; margin: 0; }}
.orig img, .unit img {{ width: 100%; height: auto; background: #ddd; border: 1px solid #ccc; }}
.units {{ display: flex; gap: 12px; flex-wrap: wrap; flex: 1; }}
.unit {{ margin: 0; flex: 1 1 280px; max-width: 520px; }}
.imgwrap {{ position: relative; display: block; }}
.order {{ position: absolute; top: 6px; left: 6px; min-width: 22px; height: 22px;
         padding: 0 6px; border-radius: 12px; background: #e6c07b; color: #1b1b1b;
         font-size: 13px; font-weight: 700; line-height: 22px; text-align: center;
         box-shadow: 0 1px 3px rgba(0,0,0,.4); box-sizing: border-box; }}
.order.fn {{ background: #20b24a; color: #fff; }}
figcaption {{ font-size: 12px; color: #555; margin-bottom: 4px; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #faf8f3;
      border: 1px solid #e6e0d4; padding: 8px; font-size: 14px; line-height: 1.45;
      max-height: 280px; overflow: auto; font-family: "Noto Serif Tibetan", "Noto Sans Tibetan", serif; }}
</style>
</head>
<body>
<nav>
  <h1>paddleocr_v2 × TDLA-v2 — {n} pages ({n_fn} with footnotes, {n_col} two-column)</h1>
  <button class="on" data-f="all">all</button>
  <button data-f="has-fn">footnotes</button>
  <button data-f="has-col">two-col</button>
  <button data-f="removed-h">removed=h</button>
  <button data-f="removed-f">removed=f</button>
  <button data-f="removed-hf">removed=hf</button>
</nav>
{''.join(cards)}
<script>
const buttons = [...document.querySelectorAll('nav button')];
const cards = [...document.querySelectorAll('.card')];
buttons.forEach(b => b.onclick = () => {{
  buttons.forEach(x => x.classList.toggle('on', x === b));
  const f = b.dataset.f;
  cards.forEach(c => {{
    c.classList.toggle('hidden', f !== 'all' && !c.dataset.filter.includes(f));
  }});
}});
</script>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/paddleocr_v2_gallery"))
    p.add_argument("--ds-dir", type=Path, default=Path("/tmp/tdla-v2"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.out_dir / "img"
    img_dir.mkdir(exist_ok=True)

    if not args.skip_download:
        need = not (args.ds_dir / "labels" / "val").is_dir() and not (
            args.ds_dir / "labels" / "test"
        ).is_dir()
        if need:
            print("downloading TDLA-v2 labels…", flush=True)
            _download_labels(args.ds_dir)
        else:
            print(f"using existing dataset at {args.ds_dir}", flush=True)

    pages = _iter_label_pages(args.ds_dir)
    print(f"indexed {len(pages)} val/test pages", flush=True)
    sample = _sample_pages(pages, args.limit, args.seed)
    print(
        "sample buckets: "
        + ", ".join(
            f"{k}={sum(1 for x in sample if x['bucket']==k)}"
            for k in ("fn+cols", "footnote", "two-col", "hf", "h", "f", "many-ta", "plain")
        ),
        flush=True,
    )
    if not args.skip_download:
        missing = [s["rel"] for s in sample if not (args.ds_dir / s["rel"]).exists()]
        if missing:
            print(f"downloading {len(missing)} images…", flush=True)
            _download_images(args.ds_dir, missing)

    cfg = PaddleOCRConfig(
        layout_mask_enabled=True,
        layout_split_columns=True,
        layout_isolate_footnotes=True,
        ocr_batch_size=8,
    )
    worker = None
    if not args.skip_ocr:
        from bec_orch.jobs.paddleocr.worker import PaddleOCRJobWorker

        print(f"loading engine from {cfg.resolved_model_dir()}", flush=True)
        worker = PaddleOCRJobWorker(cfg)

    records: list[dict] = []
    for i, spec in enumerate(sample):
        src = args.ds_dir / spec["rel"]
        print(f"[{i+1}/{len(sample)}] {spec['stem']} {spec['tags']}", flush=True)
        data = src.read_bytes()
        rgb, res_scale = bytes_to_rgb(data, cfg)
        orig = _annotate(rgb, spec["boxes"])
        orig_rel = f"img/{spec['stem']}_orig.jpg"
        _save_jpeg(orig, args.out_dir / orig_rel)

        units, n_masked, n_columns, n_fn = _v2_units(rgb, spec["boxes"], cfg)
        texts = [""] * len(units)
        if worker is not None:
            texts = _ocr_images(worker, [u["image"] for u in units])
        unit_recs = []
        for u, text in zip(units, texts, strict=True):
            rel = f"img/{spec['stem']}_{u['kind']}{u['index']}.jpg"
            _save_jpeg(u["image"], args.out_dir / rel, max_side=2000)
            unit_recs.append(
                {"kind": u["kind"], "index": u["index"], "file": rel, "text": text}
            )
            u["image"].close()
        rgb.close()
        orig.close()
        rec = {
            "stem": spec["stem"],
            "split": spec["split"],
            "rel": spec["rel"],
            "tags": spec["tags"],
            "bucket": spec["bucket"],
            "removed": spec["removed"],
            "res_scale": res_scale,
            "n_masked_boxes": n_masked,
            "n_columns": n_columns,
            "n_footnotes": n_fn,
            "n_units": len(unit_recs),
            "original": orig_rel,
            "units": unit_recs,
        }
        records.append(rec)
        (args.out_dir / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    html_path = _write_html(args.out_dir, records)
    print(f"wrote {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
