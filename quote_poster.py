"""
quote_poster.py — 名言画像 → GitHub Pages → Pinterest 自動投稿パイプライン

使い方:
    python quote_poster.py
    python quote_poster.py --dry-run   # Pinterest への投稿をスキップ
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv(Path(__file__).parent / ".env")

# ── 環境変数 ────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
HF_API_TOKEN         = os.getenv("HF_API_TOKEN", "")
PINTEREST_TOKEN      = os.getenv("PINTEREST_ACCESS_TOKEN", "")
PINTEREST_BOARD_ID   = os.getenv("PINTEREST_BOARD_ID", "")
GITHUB_PAGES_BASE    = os.getenv("PAGES_BASE_URL", "").rstrip("/")
GITHUB_REPO          = os.getenv("GITHUB_REPOSITORY", "")   # owner/repo

# HF モデル（無料の Inference API で動く軽量モデル）
HF_IMAGE_MODEL = os.getenv(
    "HF_IMAGE_MODEL",
    "black-forest-labs/FLUX.1-schnell",
)
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DOCS_DIR   = Path(__file__).parent / "docs" / "pins"
FONT_PATH  = Path(__file__).parent / "fonts" / "Inter-Bold.ttf"
OUTPUT_DIR = Path(__file__).parent / "output" / "quotes"

# ── ロガー ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("quote_poster")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1 — OpenRouter で名言テキスト生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_quote() -> dict[str, str]:
    """
    OpenRouter (無料モデル) で英語の名言を1つ生成し、
    {"quote": "...", "author": "...", "image_prompt": "..."} を返す。
    """
    log.info("[Step1] OpenRouter で名言を生成中…")

    prompt = textwrap.dedent("""\
        Generate ONE original English motivational quote (under 20 words).
        Then write a vivid cinematic image prompt (under 30 words) that visually
        represents the emotion of the quote — no text in the image.

        Respond ONLY with a JSON object, no markdown fences:
        {
          "quote": "<the motivational quote>",
          "author": "<real or fictional author name>",
          "image_prompt": "<stable diffusion prompt for the background image>"
        }
    """)

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "nex-agi/nex-n2-pro:free",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # マークダウンコードブロックが混入した場合に除去
    raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()

    data = json.loads(raw)
    log.info("[Step1] 名言: %s — %s", data["quote"], data["author"])
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2 — Hugging Face で背景画像生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_background(image_prompt: str) -> bytes:
    """
    Hugging Face Inference API で背景画像を生成し、PNG バイト列を返す。
    モデルがロード中（503）の場合は最大 5 分待機してリトライする。
    """
    log.info("[Step2] HF Inference API で背景画像を生成中…")
    log.info("[Step2] prompt: %s", image_prompt)

    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {
        "inputs": image_prompt + ", ultra-detailed, 4k, cinematic lighting, no text",
        "parameters": {"width": 1024, "height": 1024},
    }

    for attempt in range(1, 11):
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

        if resp.status_code == 200:
            # レスポンスが JSON（base64）か生バイナリかを判定
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                data = resp.json()
                if isinstance(data, list) and "generated_image" in data[0]:
                    return base64.b64decode(data[0]["generated_image"])
                # FLUX 系は生バイナリを返すこともある
            return resp.content

        if resp.status_code == 503:
            wait = min(30 * attempt, 120)
            log.warning("[Step2] モデルロード中 (503)… %d 秒後リトライ (attempt %d/10)", wait, attempt)
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise TimeoutError("Hugging Face モデルが起動しませんでした（10 回リトライ）")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3 — Pillow でテキスト合成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    # フォールバック: システムフォント
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def composite_text(bg_bytes: bytes, quote: str, author: str) -> bytes:
    """背景画像の上に名言テキストを合成して PNG バイト列を返す。"""
    log.info("[Step3] テキストを合成中…")

    img = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 半透明グラデーション帯（下半分）
    for y in range(H // 2, H):
        alpha = int(180 * (y - H // 2) / (H // 2))
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # 名言テキスト（折り返し）
    quote_font_size = max(36, W // 20)
    author_font_size = max(24, W // 32)
    quote_font  = _load_font(quote_font_size)
    author_font = _load_font(author_font_size)

    # テキスト折り返し（最大 30 文字/行）
    wrapped = textwrap.fill(f'"{quote}"', width=30)
    lines = wrapped.split("\n")

    line_h = quote_font_size + 10
    total_text_h = line_h * len(lines) + author_font_size + 20
    y_start = H - total_text_h - int(H * 0.08)

    margin = int(W * 0.07)

    for i, line in enumerate(lines):
        y = y_start + i * line_h
        # 影
        draw.text((margin + 2, y + 2), line, font=quote_font, fill=(0, 0, 0, 180))
        draw.text((margin, y), line, font=quote_font, fill=(255, 255, 255, 240))

    # 著者名
    author_y = y_start + len(lines) * line_h + 12
    draw.text((margin + 2, author_y + 2), f"— {author}", font=author_font, fill=(0, 0, 0, 160))
    draw.text((margin, author_y), f"— {author}", font=author_font, fill=(220, 220, 220, 220))

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    log.info("[Step3] 合成完了 (%d bytes)", buf.tell())
    return buf.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4 — GitHub Pages クッションページを生成してコミット
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta property="og:title" content="{quote}">
  <meta property="og:description" content="— {author}">
  <meta property="og:image" content="{image_url}">
  <meta property="og:type" content="website">
  <title>{quote}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Georgia, serif;
      background: #0a0a0a;
      color: #f5f5f5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }}
    .card {{
      max-width: 700px;
      width: 100%;
      text-align: center;
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      box-shadow: 0 8px 40px rgba(0,0,0,.6);
      margin-bottom: 2rem;
    }}
    blockquote {{
      font-size: clamp(1.1rem, 3vw, 1.6rem);
      line-height: 1.7;
      font-style: italic;
      color: #eee;
      margin-bottom: 1rem;
    }}
    cite {{
      font-size: 1rem;
      color: #aaa;
    }}
    .ts {{
      margin-top: 2rem;
      font-size: .75rem;
      color: #555;
    }}
  </style>
</head>
<body>
  <div class="card">
    <img src="{image_url}" alt="Quote image" loading="lazy">
    <blockquote>&ldquo;{quote}&rdquo;</blockquote>
    <cite>&mdash; {author}</cite>
    <p class="ts">Generated {timestamp}</p>
  </div>
</body>
</html>
"""


def build_cushion_page(
    slug: str,
    quote: str,
    author: str,
    image_jpg_bytes: bytes,
) -> tuple[str, Path]:
    """
    docs/pins/{slug}.html と docs/pins/{slug}.jpg を書き出す。
    GitHub Pages 上の URL を返す。
    """
    log.info("[Step4] クッションページを生成中…")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    img_path  = DOCS_DIR / f"{slug}.jpg"
    html_path = DOCS_DIR / f"{slug}.html"

    img_path.write_bytes(image_jpg_bytes)

    # GitHub Pages URL
    if GITHUB_PAGES_BASE:
        image_url = f"{GITHUB_PAGES_BASE}/pins/{slug}.jpg"
        page_url  = f"{GITHUB_PAGES_BASE}/pins/{slug}.html"
    else:
        # ローカルテスト用フォールバック
        image_url = f"https://example.github.io/repo/pins/{slug}.jpg"
        page_url  = f"https://example.github.io/repo/pins/{slug}.html"

    html = _HTML_TEMPLATE.format(
        quote=quote.replace('"', "&quot;"),
        author=author,
        image_url=image_url,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    html_path.write_text(html, encoding="utf-8")

    log.info("[Step4] 書き出し完了: %s", html_path)
    return page_url, img_path


def git_commit_and_push(slug: str) -> None:
    """生成したファイルを git でコミット & プッシュする。"""
    log.info("[Step4] git commit & push…")

    def run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"コマンド失敗: {' '.join(cmd)}\n{result.stderr}")
        if result.stdout.strip():
            log.debug(result.stdout.strip())

    run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
    run(["git", "config", "user.name", "github-actions[bot]"])
    run(["git", "add", str(DOCS_DIR / f"{slug}.html"), str(DOCS_DIR / f"{slug}.jpg")])
    run(["git", "commit", "-m", f"chore: auto pin {slug}"])
    run(["git", "push"])

    log.info("[Step4] プッシュ完了")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5 — Pinterest API で投稿
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def post_to_pinterest(
    quote: str,
    author: str,
    page_url: str,
    image_url: str,
) -> str:
    """
    Pinterest v5 API でピンを作成し、作成された pin_id を返す。
    image_url は GitHub Pages 上の画像への直リンクを使う。
    """
    log.info("[Step5] Pinterest へ投稿中…")

    if not PINTEREST_TOKEN:
        raise EnvironmentError("PINTEREST_ACCESS_TOKEN が設定されていません")
    if not PINTEREST_BOARD_ID:
        raise EnvironmentError("PINTEREST_BOARD_ID が設定されていません")

    # GitHub Pages が反映されるまで少し待機（初回コミット後は特に必要）
    log.info("[Step5] GitHub Pages の反映を 30 秒待機…")
    time.sleep(30)

    body = {
        "board_id": PINTEREST_BOARD_ID,
        "title": quote[:100],
        "description": f'"{quote}" — {author}',
        "link": page_url,
        "media_source": {
            "source_type": "image_url",
            "url": image_url,
        },
    }

    resp = requests.post(
        "https://api.pinterest.com/v5/pins",
        headers={
            "Authorization": f"Bearer {PINTEREST_TOKEN}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    resp.raise_for_status()

    pin_id: str = resp.json().get("id", "unknown")
    log.info("[Step5] ピン作成完了: pin_id=%s", pin_id)
    return pin_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(dry_run: bool = False) -> None:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = f"pin_{ts}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 名言生成
    quote_data    = generate_quote()
    quote         = quote_data["quote"]
    author        = quote_data["author"]
    image_prompt  = quote_data["image_prompt"]

    # Step 2: 背景画像生成
    bg_bytes = generate_background(image_prompt)
    (OUTPUT_DIR / f"{slug}_bg.jpg").write_bytes(bg_bytes)

    # Step 3: テキスト合成
    final_jpg = composite_text(bg_bytes, quote, author)
    (OUTPUT_DIR / f"{slug}_final.jpg").write_bytes(final_jpg)
    log.info("[Step3] ローカル保存: output/quotes/%s_final.jpg", slug)

    # Step 4: クッションページ生成 & git push
    page_url, _ = build_cushion_page(slug, quote, author, final_jpg)

    if not dry_run:
        git_commit_and_push(slug)

        # Step 5: Pinterest 投稿
        image_url = f"{GITHUB_PAGES_BASE}/pins/{slug}.jpg" if GITHUB_PAGES_BASE else ""
        pin_id = post_to_pinterest(quote, author, page_url, image_url)
        log.info("完了! pin_id=%s  page=%s", pin_id, page_url)
    else:
        log.info("[dry-run] Pinterest 投稿と git push をスキップしました")
        log.info("[dry-run] page_url=%s", page_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="名言 → 画像 → Pinterest 自動投稿")
    parser.add_argument("--dry-run", action="store_true", help="投稿・push をスキップ")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
