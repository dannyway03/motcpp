#!/usr/bin/env python3
"""
Generate a polished LinkedIn demo video for motcpp.

Creates a 1280×720 ~32s video: title card → 2×2 split-screen tracking → outro.
Sequences loop seamlessly so there are no blank frames.
All text is rendered with PIL + DejaVu Sans (proper Unicode, sharp glyphs).

Usage:
    python scripts/generate_linkedin_demo.py
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Fonts ────────────────────────────────────────────────────────────────────
FONT_DIR  = Path("/usr/share/fonts/truetype/dejavu")
FONT_REG  = str(FONT_DIR / "DejaVuSans.ttf")
FONT_BOLD = str(FONT_DIR / "DejaVuSans-Bold.ttf")

def load_fonts():
    return {
        "title":    ImageFont.truetype(FONT_BOLD, 72),
        "h2":       ImageFont.truetype(FONT_BOLD, 32),
        "h3":       ImageFont.truetype(FONT_BOLD, 22),
        "body":     ImageFont.truetype(FONT_REG,  20),
        "small":    ImageFont.truetype(FONT_REG,  16),
        "tiny":     ImageFont.truetype(FONT_REG,  13),
        "hud_name": ImageFont.truetype(FONT_BOLD, 20),
        "hud_stat": ImageFont.truetype(FONT_REG,  14),
        "badge":    ImageFont.truetype(FONT_BOLD, 13),
        "track_id": ImageFont.truetype(FONT_BOLD, 13),
    }

FONTS = load_fonts()

# ─── Config ───────────────────────────────────────────────────────────────────
ASSETS_BASE = Path("/media/abi/ext_nvme2/boxmot/motcpp/assets/MOT17-ablation/train")
OUTPUT_DIR  = Path("/media/abi/ext_nvme2/boxmot/motcpp/docs/images")

SEQUENCES = {
    "ByteTrack":   ("MOT17-09", "1100 FPS  |  HOTA 66.5"),
    "OC-SORT":     ("MOT17-02", " 850 FPS  |  HOTA 64.6"),
    "BoostTrack":  ("MOT17-05", "  75 FPS  |  HOTA 67.5  ★ SOTA"),
    "OracleTrack": ("MOT17-10", " 449 FPS  |  HOTA 66.9"),
}

OUT_W, OUT_H = 1280, 720
CELL_W, CELL_H = OUT_W // 2, OUT_H // 2
FPS        = 30
INTRO_SEC  = 3.0
TRACK_SEC  = 25.0
OUTRO_SEC  = 3.5
TRAJ_LEN   = 40

# BGR colours (OpenCV)
C_BG    = (15,  15,  20)
C_CYAN  = (255, 210,  0)    # BGR: openCV uses BGR
C_WHITE = (240, 240, 240)
C_GREY  = (150, 150, 160)
C_GREEN = ( 80, 210,  60)
C_GOLD  = ( 50, 200, 255)
C_BAR   = ( 10,  10,  12)

# PIL colours (RGB)
P_BG    = ( 15,  15,  20)
P_CYAN  = (  0, 210, 255)
P_WHITE = (240, 240, 240)
P_GREY  = (150, 150, 160)
P_GREEN = ( 60, 210,  80)
P_GOLD  = (255, 200,  50)
P_DARK  = ( 10,  10,  12)


# ─── PIL helpers ──────────────────────────────────────────────────────────────

def bgr2pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil2bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple:
    """Return (width, height) of text bounding box."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def put_text(canvas_bgr: np.ndarray,
             text: str,
             xy: tuple,
             font: ImageFont.FreeTypeFont,
             color_rgb: tuple,
             alpha: float = 1.0) -> np.ndarray:
    """Draw a single text string onto a BGR numpy array via PIL."""
    pil = bgr2pil(canvas_bgr)
    draw = ImageDraw.Draw(pil)
    if alpha < 1.0:
        # Draw on separate layer and blend
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer)
        d2.text(xy, text, font=font, fill=(*color_rgb, int(alpha * 255)))
        pil = Image.alpha_composite(pil.convert("RGBA"), layer).convert("RGB")
    else:
        draw.text(xy, text, font=font, fill=color_rgb)
    return pil2bgr(pil)


def put_texts(canvas_bgr: np.ndarray, items: list, alpha: float = 1.0) -> np.ndarray:
    """
    Batch text draw — items: [(text, (x,y), font, color_rgb), ...]
    One PIL round-trip for all text on this canvas.
    """
    pil = bgr2pil(canvas_bgr).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    a = int(alpha * 255)
    for text, xy, font, color_rgb in items:
        draw.text(xy, text, font=font, fill=(*color_rgb, a))
    out = Image.alpha_composite(pil, layer).convert("RGB")
    return pil2bgr(out)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_gt(gt_file: Path) -> dict:
    tracks = defaultdict(list)
    if not gt_file.exists():
        return tracks
    with open(gt_file) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 9:
                continue
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            cls, vis = int(p[7]), float(p[8])
            if cls not in (1, 2, 7) or vis < 0.2:
                continue
            tracks[frame].append({'id': tid, 'bbox': [x, y, x+w, y+h]})
    return tracks


def track_color(tid: int) -> tuple:
    np.random.seed(tid * 17 + 31)
    h = int(np.random.randint(0, 180))
    s = int(np.random.randint(160, 255))
    v = int(np.random.randint(190, 255))
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ─── Per-cell rendering ───────────────────────────────────────────────────────

def draw_tracks_cv(cell: np.ndarray, tracks: list, trajs: dict,
                   sx: float, sy: float) -> np.ndarray:
    """Draw trajectories and bounding boxes (pure OpenCV — fast)."""
    overlay = cell.copy()
    for t in tracks:
        tid = t['id']
        x1 = int(t['bbox'][0] * sx); y1 = int(t['bbox'][1] * sy)
        x2 = int(t['bbox'][2] * sx); y2 = int(t['bbox'][3] * sy)
        cx, cy = (x1 + x2) // 2, y2
        trajs.setdefault(tid, []).append((cx, cy))
        trajs[tid] = trajs[tid][-TRAJ_LEN:]
        col = track_color(tid)
        pts = trajs[tid]
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(overlay, pts[i-1], pts[i], col, max(1, int(2 * a)),
                     cv2.LINE_AA)
    cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
    for t in tracks:
        tid = t['id']
        col = track_color(tid)
        x1 = int(t['bbox'][0] * sx); y1 = int(t['bbox'][1] * sy)
        x2 = int(t['bbox'][2] * sx); y2 = int(t['bbox'][3] * sy)
        cv2.rectangle(cell, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    return cell


def draw_track_ids_pil(cell: np.ndarray, tracks: list,
                        sx: float, sy: float) -> np.ndarray:
    """Draw track ID badges with PIL for crisp text."""
    pil = bgr2pil(cell).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = FONTS["track_id"]
    for t in tracks:
        tid = t['id']
        col_bgr = track_color(tid)
        col_rgb = (col_bgr[2], col_bgr[1], col_bgr[0])
        x1 = int(t['bbox'][0] * sx); y1 = int(t['bbox'][1] * sy)
        label = str(tid)
        tw, th = text_size(label, font)
        pad = 3
        bx1 = x1
        by1 = max(0, y1 - th - pad * 2)
        bx2 = max(bx1 + 1, x1 + tw + pad * 2)
        by2 = max(by1 + 1, y1)
        draw.rectangle([bx1, by1, bx2, by2], fill=(*col_rgb, 210))
        draw.text((bx1 + pad, by1 + pad), label, font=font,
                  fill=(255, 255, 255, 255))
    out = Image.alpha_composite(pil, layer).convert("RGB")
    return pil2bgr(out)


def draw_cell_hud(cell: np.ndarray, name: str, stats: str,
                  n_tracks: int) -> np.ndarray:
    """HUD bar at top of each cell, rendered with PIL."""
    bar_h = 38
    # Dark bar (OpenCV — fast)
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (CELL_W, bar_h), C_BAR, -1)
    cell = cv2.addWeighted(overlay, 0.85, cell, 0.15, 0)
    cv2.line(cell, (0, bar_h), (CELL_W, bar_h), (50, 50, 58), 1)

    # Text via PIL
    items = [
        (name,    (10, 8),  FONTS["hud_name"], P_CYAN),
        (stats,   (CELL_W - text_size(stats, FONTS["hud_stat"])[0] - 8, 12),
                            FONTS["hud_stat"], P_GREY),
        (f"tracks: {n_tracks}", (10, CELL_H - 20), FONTS["tiny"], P_GREY),
    ]
    return put_texts(cell, items)


# ─── Canvas composition ───────────────────────────────────────────────────────

def compose_grid(cells: list) -> np.ndarray:
    top = np.hstack([cells[0], cells[1]])
    bot = np.hstack([cells[2], cells[3]])
    canvas = np.vstack([top, bot])
    cv2.line(canvas, (CELL_W, 0),    (CELL_W, OUT_H), (50, 50, 58), 1)
    cv2.line(canvas, (0, CELL_H),    (OUT_W, CELL_H), (50, 50, 58), 1)
    return canvas


def draw_global_hud(canvas: np.ndarray, fi: int, total: int) -> np.ndarray:
    """Bottom branding bar."""
    bar_h = 30
    h = OUT_H
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (OUT_W, h), C_BAR, -1)
    canvas = cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0)
    cv2.line(canvas, (0, h - bar_h), (OUT_W, h - bar_h), (50, 50, 58), 1)

    # Progress bar
    prog = int((fi / max(total, 1)) * 180)
    cv2.rectangle(canvas, (OUT_W - 195, h - 20), (OUT_W - 15, h - 12),
                  (40, 40, 48), -1)
    if prog:
        cv2.rectangle(canvas, (OUT_W - 195, h - 20),
                      (OUT_W - 195 + prog, h - 12), C_CYAN[::-1], -1)

    gh = "github.com/Geekgineer/motcpp"
    gh_w = text_size(gh, FONTS["tiny"])[0]
    items = [
        ("motcpp", (10, h - bar_h + 5), FONTS["hud_name"], P_CYAN),
        ("  Modern C++ Multi-Object Tracking  |  10-100x faster than Python",
         (82, h - bar_h + 7), FONTS["tiny"], P_GREY),
        (gh, (OUT_W - gh_w - 15, h - bar_h + 3), FONTS["tiny"], P_GREY),
    ]
    return put_texts(canvas, items)


# ─── Title card ───────────────────────────────────────────────────────────────

def make_title_card(t: float) -> np.ndarray:
    alpha = min(1.0, t / 0.7)
    canvas = np.full((OUT_H, OUT_W, 3), C_BG, dtype=np.uint8)

    cy = OUT_H // 2
    # Accent lines
    cv2.line(canvas, (70, cy - 68), (OUT_W - 70, cy - 68), C_CYAN[::-1], 1)
    cv2.line(canvas, (70, cy + 95), (OUT_W - 70, cy + 95), C_CYAN[::-1], 1)

    title     = "motcpp"
    tagline   = "Modern C++ Multi-Object Tracking"
    feats     = "10 SOTA algorithms  |  10-100x faster than Python  |  Production-ready"

    tw_title = text_size(title, FONTS["title"])[0]
    tw_tag   = text_size(tagline, FONTS["h3"])[0]
    tw_feat  = text_size(feats, FONTS["small"])[0]

    items = [
        (title,   ((OUT_W - tw_title) // 2, cy - 62), FONTS["title"],  P_CYAN),
        (tagline, ((OUT_W - tw_tag)   // 2, cy + 10), FONTS["h3"],     P_WHITE),
        (feats,   ((OUT_W - tw_feat)  // 2, cy + 50), FONTS["small"],  P_GREY),
    ]

    # Tracker chips
    names = list(SEQUENCES.keys())
    chip_w, chip_h, gap = 148, 30, 10
    total_chips_w = len(names) * (chip_w + gap) - gap
    sx = (OUT_W - total_chips_w) // 2
    chip_y = cy + 100

    pil = bgr2pil(canvas).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    a = int(alpha * 255)

    # Draw text items
    for text, xy, font, color_rgb in items:
        draw.text(xy, text, font=font, fill=(*color_rgb, a))

    # Draw chips
    for i, n in enumerate(names):
        x = sx + i * (chip_w + gap)
        draw.rectangle([x, chip_y, x + chip_w, chip_y + chip_h],
                       fill=(30, 30, 38, a))
        draw.rectangle([x, chip_y, x + chip_w, chip_y + chip_h],
                       outline=(*P_CYAN, a), width=1)
        nw, nh = text_size(n, FONTS["small"])
        draw.text((x + (chip_w - nw) // 2, chip_y + (chip_h - nh) // 2),
                  n, font=FONTS["small"], fill=(*P_WHITE, a))

    out = Image.alpha_composite(pil, layer).convert("RGB")
    return pil2bgr(out)


# ─── Outro card ───────────────────────────────────────────────────────────────

def make_outro_card(t: float) -> np.ndarray:
    alpha = min(1.0, t / 0.6)
    canvas = np.full((OUT_H, OUT_W, 3), C_BG, dtype=np.uint8)

    # Star field
    for i in range(35):
        np.random.seed(i * 7)
        sx = int(np.random.randint(0, OUT_W))
        sy = int(np.random.randint(0, OUT_H // 3))
        r  = int(np.random.randint(1, 3))
        cv2.circle(canvas, (sx, sy), r, (70, 70, 82), -1)

    # Layout: title block in upper-centre, highlights near bottom
    title_y  = 155   # "motcpp" title
    url_y    = title_y + 88
    star_y   = url_y  + 42
    line_y   = OUT_H - 148   # cyan separator line — near bottom
    hl_y     = line_y + 18   # highlights just below the line

    title = "motcpp"
    url   = "github.com/Geekgineer/motcpp"
    star  = "Star the repo ★  |  Contributions welcome"

    tw_title = text_size(title, FONTS["title"])[0]
    tw_url   = text_size(url, FONTS["h2"])[0]
    tw_star  = text_size(star, FONTS["body"])[0]

    highlights = [
        ("SORT",        "1250 FPS"),
        ("ByteTrack",   "1100 FPS"),
        ("OracleTrack", "449 FPS"),
        ("BoostTrack",  "HOTA 67.5 ★"),
    ]
    col_w = 200
    hl_w  = len(highlights) * col_w
    hx    = (OUT_W - hl_w) // 2

    # Top accent line
    cv2.line(canvas, (70, title_y - 18), (OUT_W - 70, title_y - 18),
             C_CYAN[::-1], 1)

    pil = bgr2pil(canvas).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    a = int(alpha * 255)

    draw.text(((OUT_W - tw_title) // 2, title_y), title,
              font=FONTS["title"], fill=(*P_CYAN, a))
    draw.text(((OUT_W - tw_url)   // 2, url_y),   url,
              font=FONTS["h2"],    fill=(*P_WHITE, a))
    draw.text(((OUT_W - tw_star)  // 2, star_y),  star,
              font=FONTS["body"],  fill=(*P_GREY,  a))

    # Separator line above highlights
    draw.line([(70, line_y), (OUT_W - 70, line_y)],
              fill=(*P_CYAN, a), width=1)

    for i, (name, val) in enumerate(highlights):
        x = hx + i * col_w + col_w // 2
        nw, _ = text_size(name, FONTS["small"])
        vw, _ = text_size(val,  FONTS["h3"])
        col = P_GOLD if "★" in val else P_GREEN
        draw.text((x - nw // 2, hl_y),      name, font=FONTS["small"],
                  fill=(*P_GREY, a))
        draw.text((x - vw // 2, hl_y + 22), val,  font=FONTS["h3"],
                  fill=(*col, a))

    out = Image.alpha_composite(pil, layer).convert("RGB")
    return pil2bgr(out)


# ─── High-quality GIF generator ───────────────────────────────────────────────

GIF_CFG = {
    "demo_bytetrack":   ("ByteTrack",   "MOT17-09", "1100 FPS  |  HOTA 66.5", 1),
    "demo_ocsort":      ("OC-SORT",     "MOT17-02", " 850 FPS  |  HOTA 64.6", 1),
    "demo_boosttrack":  ("BoostTrack",  "MOT17-05", "  75 FPS  |  HOTA 67.5  ★ SOTA", 1),
    "demo_sort":        ("SORT",        "MOT17-13", "1250 FPS  |  HOTA 62.4", 1),
}
GIF_W, GIF_H   = 640, 360    # output size
GIF_FPS        = 25
GIF_FRAMES     = 180          # 7.2 s of content
GIF_START      = 30           # skip first 30 frames (often static)


def make_gif(name: str, tracker: str, seq_name: str, stats: str,
             start_frame: int = GIF_START) -> None:
    seq_path = ASSETS_BASE / seq_name
    if not seq_path.exists():
        print(f"  ✗ {seq_name} not found — skipping {name}.gif")
        return

    imgs = sorted((seq_path / "img1").glob("*.jpg"))
    if not imgs:
        imgs = sorted((seq_path / "img1").glob("*.png"))
    gt = load_gt(seq_path / "gt" / "gt.txt")

    tmp_mp4 = OUTPUT_DIR / f"_{name}_tmp.mp4"
    out_gif = OUTPUT_DIR / f"{name}.gif"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(tmp_mp4), fourcc, GIF_FPS, (GIF_W, GIF_H))

    trajs: dict = {}
    bar_h = 36

    for fi in range(GIF_FRAMES):
        idx      = (start_frame + fi) % len(imgs)
        frame_no = idx + 1
        raw = cv2.imread(str(imgs[idx]))
        if raw is None:
            continue

        src_h, src_w = raw.shape[:2]
        sx = GIF_W / src_w
        sy = GIF_H / src_h
        cell = cv2.resize(raw, (GIF_W, GIF_H), interpolation=cv2.INTER_LINEAR)

        gt_tracks = gt.get(frame_no, [])
        if idx == 0:
            trajs = {}

        cell = draw_tracks_cv(cell, gt_tracks, trajs, sx, sy)
        cell = draw_track_ids_pil(cell, gt_tracks, sx, sy)

        # ── HUD bar ──
        overlay = cell.copy()
        cv2.rectangle(overlay, (0, 0), (GIF_W, bar_h), C_BAR, -1)
        cell = cv2.addWeighted(overlay, 0.85, cell, 0.15, 0)
        cv2.line(cell, (0, bar_h), (GIF_W, bar_h), (50, 50, 58), 1)

        # ── Bottom branding bar ──
        bot_h = 24
        cv2.rectangle(cell, (0, GIF_H - bot_h), (GIF_W, GIF_H), C_BAR, -1)
        cv2.line(cell, (0, GIF_H - bot_h), (GIF_W, GIF_H - bot_h), (50, 50, 58), 1)

        # ── PIL text ──
        gh = "github.com/Geekgineer/motcpp"
        gh_w = text_size(gh, FONTS["tiny"])[0]
        badge = f"tracks: {len(gt_tracks)}"
        items = [
            (tracker,  (10, 8),                      FONTS["hud_name"], P_CYAN),
            (stats,    (GIF_W - text_size(stats, FONTS["hud_stat"])[0] - 8, 12),
                                                      FONTS["hud_stat"], P_GREY),
            (badge,    (10, GIF_H - bot_h + 5),      FONTS["tiny"],     P_GREY),
            ("motcpp", (GIF_W // 2 - text_size("motcpp", FONTS["tiny"])[0] // 2,
                        GIF_H - bot_h + 5),           FONTS["tiny"],     P_CYAN),
            (gh,       (GIF_W - gh_w - 8, GIF_H - bot_h + 5),
                                                      FONTS["tiny"],     P_GREY),
        ]
        cell = put_texts(cell, items)
        writer.write(cell)

    writer.release()

    # ffmpeg: mp4 → optimised GIF with palette
    ret = os.system(
        f'ffmpeg -y -i "{tmp_mp4}" '
        f'-vf "fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos,'
        f'split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" '
        f'-loop 0 "{out_gif}" 2>/dev/null'
    )
    tmp_mp4.unlink(missing_ok=True)
    if ret == 0:
        mb = out_gif.stat().st_size / 1e6
        print(f"  ✓ {out_gif.name}  ({mb:.1f} MB)")
    else:
        print(f"  ✗ ffmpeg failed for {name}")


def generate_readme_gifs() -> None:
    print("\nGenerating README GIFs...")
    for name, (tracker, seq, stats, start) in GIF_CFG.items():
        print(f"  Rendering {name}.gif  [{tracker} / {seq}]")
        make_gif(name, tracker, seq, stats, start_frame=start)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_mp4   = OUTPUT_DIR / "linkedin_demo_raw.mp4"
    final_mp4 = OUTPUT_DIR / "linkedin_demo_final.mp4"

    # ── Load sequences ────────────────────────────────────────────────────────
    print("Loading sequences...")
    seq_data = {}
    for tracker_name, (seq_name, _) in SEQUENCES.items():
        seq_path = ASSETS_BASE / seq_name
        if not seq_path.exists():
            print(f"  ✗ {seq_name} not found")
            seq_data[tracker_name] = None
            continue
        imgs = sorted((seq_path / "img1").glob("*.jpg"))
        if not imgs:
            imgs = sorted((seq_path / "img1").glob("*.png"))
        gt = load_gt(seq_path / "gt" / "gt.txt")
        seq_data[tracker_name] = {"images": imgs, "gt": gt}
        print(f"  ✓ {seq_name}: {len(imgs)} frames  (loops every "
              f"{len(imgs)/FPS:.1f}s)")

    # ── Writer ────────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(raw_mp4), fourcc, FPS, (OUT_W, OUT_H))
    if not writer.isOpened():
        print("Error: could not open VideoWriter"); sys.exit(1)

    trackers = list(SEQUENCES.keys())
    trajs    = {n: {} for n in trackers}

    # ── Intro ─────────────────────────────────────────────────────────────────
    intro_frames = int(INTRO_SEC * FPS)
    print(f"Rendering intro ({INTRO_SEC:.0f}s, {intro_frames} frames)...")
    for f in range(intro_frames):
        writer.write(make_title_card(f / FPS))

    # ── Tracking content ──────────────────────────────────────────────────────
    track_frames = int(TRACK_SEC * FPS)
    print(f"Rendering tracking ({TRACK_SEC:.0f}s, {track_frames} frames, "
          f"sequences loop seamlessly)...")
    t0 = time.time()

    for fi in range(track_frames):
        cells = []
        for tracker_name in trackers:
            data = seq_data[tracker_name]
            _, stats_str = SEQUENCES[tracker_name]

            if data is None:
                blank = np.full((CELL_H, CELL_W, 3), C_BG, dtype=np.uint8)
                cells.append(blank)
                continue

            # Loop the sequence
            frame_idx = fi % len(data["images"])
            frame_no  = frame_idx + 1  # 1-indexed GT lookup

            raw = cv2.imread(str(data["images"][frame_idx]))
            if raw is None:
                cells.append(np.full((CELL_H, CELL_W, 3), C_BG, dtype=np.uint8))
                continue

            src_h, src_w = raw.shape[:2]
            sx = CELL_W / src_w
            sy = CELL_H / src_h
            cell = cv2.resize(raw, (CELL_W, CELL_H), interpolation=cv2.INTER_LINEAR)

            gt_tracks = data["gt"].get(frame_no, [])

            # Reset trajectories on loop boundary
            if frame_idx == 0:
                trajs[tracker_name] = {}

            cell = draw_tracks_cv(cell, gt_tracks, trajs[tracker_name], sx, sy)
            cell = draw_track_ids_pil(cell, gt_tracks, sx, sy)
            cell = draw_cell_hud(cell, tracker_name, stats_str, len(gt_tracks))
            cells.append(cell)

        canvas = compose_grid(cells)
        canvas = draw_global_hud(canvas, fi, track_frames)
        writer.write(canvas)

        if fi % 90 == 0:
            elapsed = time.time() - t0
            rfps = (fi + 1) / max(elapsed, 1e-3)
            eta  = (track_frames - fi) / max(rfps, 1e-3)
            print(f"  [{fi+1:4d}/{track_frames}]  render {rfps:.0f} fps  "
                  f"ETA {eta:.0f}s")

    # ── Outro ─────────────────────────────────────────────────────────────────
    outro_frames = int(OUTRO_SEC * FPS)
    print(f"Rendering outro ({OUTRO_SEC:.1f}s, {outro_frames} frames)...")
    for f in range(outro_frames):
        writer.write(make_outro_card(f / FPS))

    writer.release()
    print(f"\nRaw MP4 saved: {raw_mp4}")

    # ── Re-encode with ffmpeg (H.264, faststart, LinkedIn-ready) ──────────────
    total_s = INTRO_SEC + TRACK_SEC + OUTRO_SEC
    print(f"Re-encoding to H.264 (total {total_s:.0f}s)...")
    ret = os.system(
        f'ffmpeg -y -i "{raw_mp4}" '
        f'-c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p '
        f'-movflags +faststart '
        f'"{final_mp4}" 2>/dev/null'
    )
    if ret == 0:
        mb = final_mp4.stat().st_size / 1e6
        print(f"\n✓  {final_mp4}")
        print(f"   {OUT_W}x{OUT_H} @ {FPS} fps  |  {total_s:.0f}s  |  {mb:.1f} MB")
        print(f"\nLinkedIn tips:")
        print(f"  Upload linkedin_demo_final.mp4 directly.")
        print(f"  Add caption: 'motcpp — 10 SOTA tracking algorithms in C++. "
              f"Open source: github.com/Geekgineer/motcpp'")
    else:
        print(f"ffmpeg failed — raw video at: {raw_mp4}")

    # ── README GIFs ───────────────────────────────────────────────────────────
    generate_readme_gifs()


if __name__ == "__main__":
    main()
