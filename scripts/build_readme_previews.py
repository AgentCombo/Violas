"""Rebuild lightweight README case previews. Requires Pillow (docs only)."""
from pathlib import Path
import re

from PIL import Image, ImageOps


def build(repo):
    readme = (repo / 'README.md').read_text(encoding='utf8')
    # Full-resolution links remain the source of truth on subsequent rebuilds.
    sources = sorted(set(re.findall(r'href="(docs/figures/readme/case-[^"]+\.jpg)"', readme)))
    if not sources:
        raise ValueError('No case image links found in README.md')
    old_bytes = new_bytes = 0
    for relative in sources:
        source = repo / relative
        output = source.parent / 'previews' / source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            # 2x pixel density for a 360px-wide README card.
            width = min(720, image.width)
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            image.save(output, 'JPEG', quality=85, subsampling=0,
                       optimize=True, progressive=True)
        old_bytes += source.stat().st_size
        new_bytes += output.stat().st_size
    print(f'{len(sources)} previews: {old_bytes:,} -> {new_bytes:,} bytes '
          f'({100 * (1 - new_bytes / old_bytes):.1f}% smaller)')


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
