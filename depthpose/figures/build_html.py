"""Build a self-contained dist/index.html from reports/blog.md.

What this does:

1. Convert markdown to HTML using `markdown` (with footnotes + tables + fenced_code).
2. Wrap the body in a minimal styled HTML document so it renders cleanly
   without internet access (no remote CSS, no remote fonts, no remote scripts).
3. Replace the static `video_grid.png` figure reference with a 1×3 grid of
   embedded `<video controls preload="metadata">` tags pointing to the
   relevant mp4s.
4. Italic paragraphs that immediately follow an image are styled as figure
   captions.
5. Copy the referenced figures + the three videos into `dist/figures/` and
   `dist/videos/` so the directory is self-contained when zipped.

Usage::

    python -m depthpose.figures.build_html

Output: ``dist/index.html`` + ``dist/figures/`` + ``dist/videos/``.
"""
from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path

import markdown


def _transcode_to_h264(src: Path, dst: Path) -> None:
    """Re-encode an OpenCV-written mp4 (FOURCC mp4v / FMP4) into H.264 + yuv420p
    + faststart, so it plays in <video> tags in Chrome/Safari/Firefox.

    Uses the bundled ffmpeg from imageio-ffmpeg so we don't depend on a
    system ffmpeg install.
    """
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ff, "-y", "-loglevel", "error",
         "-i", str(src),
         "-c:v", "libx264",
         "-pix_fmt", "yuv420p",
         "-preset", "fast",
         "-crf", "23",
         "-movflags", "+faststart",
         str(dst)],
        check=True,
    )

ROOT = Path("/home/farandhigh-ubuntu/Documents/cv/depth-pose-tracking")
SRC = ROOT / "reports" / "blog.md"
DIST = ROOT / "dist"

# Two videos to embed for the §7 "Watching the model run" grid.
VIDEO_GRID = [
    ("S01_7.mp4",          "Best case: rs_blue_car_light_change",      "8 mm"),
    ("S01_14_holdout.mp4", "Held-out: rs_up_incline (S01/14)",         "40 mm"),
]


def main() -> None:
    DIST.mkdir(exist_ok=True)
    (DIST / "figures").mkdir(exist_ok=True)
    (DIST / "videos").mkdir(exist_ok=True)

    md_text = SRC.read_text()

    # --- Convert markdown body ---
    md = markdown.Markdown(extensions=["footnotes", "tables", "fenced_code", "smarty"])
    body_html = md.convert(md_text)

    # --- Style italic paragraphs that follow an <img> as figure captions ---
    # Pattern: <p><img ...></p>\n<p><em>...</em></p>
    body_html = re.sub(
        r'<p>(<img[^>]+>)</p>\s*<p><em>(.*?)</em></p>',
        r'<figure>\1<figcaption><em>\2</em></figcaption></figure>',
        body_html,
        flags=re.DOTALL,
    )

    # --- Replace the video_grid.png figure with embedded <video> grid ---
    video_html_parts = ['<figure class="video-grid"><div class="video-row">']
    for fname, label, mpjpe in VIDEO_GRID:
        video_html_parts.append(
            f'<div class="video-cell">'
            f'<video controls preload="metadata"><source src="videos/{fname}" type="video/mp4">'
            f'Your browser does not support the video tag.</video>'
            f'<div class="video-label">{label}<br><span class="video-mpjpe">MPJPE vs oracle: {mpjpe}</span></div>'
            f'</div>'
        )
    video_html_parts.append('</div>')
    # Caption (re-uses the figure-grid caption from the markdown).
    video_html_parts.append(
        '<figcaption><em>Two side-by-side recordings: best in-distribution case '
        '(<code>rs_blue_car_light_change</code>, 8 mm MPJPE under the random '
        'split) and the truly held-out bag (<code>rs_up_incline</code>, 40 mm '
        '— the bag the model never saw during training). Oracle (green skeleton '
        'on RGB) on the left of each panel, student (cyan/magenta on depth) on '
        'the right. Each panel carries a live HUD with the running cadence '
        '(steps/min) and stride period (s) per side, computed causally from '
        'ankle-z peak detection over the frames seen so far — the oracle reads '
        'the parquet labels, the student its own output, so the two HUDs '
        'visibly converge to the same gait numbers as the recording plays.'
        '</em></figcaption></figure>'
    )
    video_html = "".join(video_html_parts)

    # The video_grid.png figure block in the HTML — replace whole <figure>
    body_html = re.sub(
        r'<figure><img[^>]+video_grid\.png[^>]*>\s*<figcaption>.*?</figcaption></figure>',
        video_html, body_html, count=1, flags=re.DOTALL,
    )
    # Fallback: in case the figure transform didn't match
    body_html = re.sub(
        r'<p><img[^>]+video_grid\.png[^>]*></p>',
        video_html, body_html, count=1,
    )

    # --- Wrap in a minimal styled document ---
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>depthpose: a 5 MB depth-only model for 3D lower-body pose tracking on a walker</title>
<style>
  :root {{
    --max: 760px;
    --fg: #1a1a1a;
    --muted: #555;
    --accent: #1a4f8a;
    --rule: #d8d8d8;
    --bg: #fdfdfb;
    --code-bg: #f3f3ef;
  }}
  html {{ background: var(--bg); }}
  body {{
    font-family: 'Source Serif Pro', 'Charter', Georgia, serif;
    color: var(--fg);
    line-height: 1.55;
    max-width: var(--max);
    margin: 2.5em auto 6em;
    padding: 0 1.2em;
    font-size: 16.5px;
  }}
  h1 {{ font-size: 1.8em; line-height: 1.2; margin-bottom: 0.1em; }}
  h2 {{ margin-top: 2.2em; padding-top: 0.4em; border-top: 1px solid var(--rule); font-size: 1.35em; }}
  h3 {{ font-size: 1.1em; }}
  p {{ margin: 0.9em 0; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{
    font-family: 'Source Code Pro', 'Menlo', Consolas, monospace;
    background: var(--code-bg);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.92em;
  }}
  pre {{ background: var(--code-bg); padding: 0.9em 1em; border-radius: 4px; overflow-x: auto; }}
  pre code {{ background: transparent; padding: 0; }}
  figure {{ margin: 1.6em 0; text-align: center; }}
  figure img {{ max-width: 100%; height: auto; }}
  figcaption {{
    color: var(--muted);
    font-size: 0.92em;
    margin-top: 0.5em;
    text-align: left;
  }}
  figcaption em {{ font-style: italic; }}
  table {{ border-collapse: collapse; margin: 1.2em 0; font-size: 0.94em; }}
  table th, table td {{ border: 1px solid var(--rule); padding: 4px 9px; text-align: left; }}
  table th {{ background: #f3f3ef; }}
  blockquote {{ border-left: 3px solid var(--accent); padding-left: 1em; color: var(--muted); margin: 1em 0; }}

  /* Video grid */
  .video-grid .video-row {{ display: flex; gap: 0.8em; flex-wrap: wrap; justify-content: space-between; }}
  .video-cell {{ flex: 1 1 31%; min-width: 200px; }}
  .video-cell video {{ width: 100%; height: auto; background: #000; }}
  .video-label {{ font-size: 0.85em; color: var(--fg); margin-top: 0.25em; text-align: left; }}
  .video-mpjpe {{ color: var(--muted); }}

  /* Footnotes */
  .footnote {{ font-size: 0.92em; color: var(--muted); }}
  .footnote ol {{ padding-left: 1.6em; }}
  .footnote li {{ margin-bottom: 0.45em; }}
  hr.footnote-sep {{ display: none; }}

  /* Author / date line */
  body > p:nth-of-type(1) em {{ color: var(--muted); }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    out = DIST / "index.html"
    out.write_text(html)

    # --- Copy figures referenced in the markdown ---
    for img in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', md_text):
        # img path is relative to reports/ directory
        src = (ROOT / "reports" / img).resolve()
        if src.exists():
            dst = DIST / Path(img)   # preserves "figures/X.png" structure
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    # --- Transcode the three referenced videos to H.264 ---
    # OpenCV's VideoWriter writes mp4v (MPEG-4 Part 2), which Chrome/Safari/
    # Firefox don't play in <video> tags. We re-encode to H.264 (avc1) +
    # yuv420p + faststart on each build. Side benefit: H.264 is ~3.5×
    # smaller than mp4v at visually-equivalent quality.
    for fname, _, _ in VIDEO_GRID:
        src = ROOT / "reports" / "videos" / fname
        dst = DIST / "videos" / fname
        if not src.exists():
            print(f"warning: missing source {src}")
            continue
        _transcode_to_h264(src, dst)

    print(f"wrote {out.relative_to(ROOT)}")
    print(f"figures: {len(list((DIST / 'figures').iterdir()))}")
    print(f"videos:  {len(list((DIST / 'videos').iterdir()))}")
    print(f"\nzip the {DIST.name}/ directory for submission.")


if __name__ == "__main__":
    main()
