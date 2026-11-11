#!/usr/bin/env python3
"""Prepare semantic diversity facets for an image benchmark.

The first version is deliberately small and resumable:

1. ``discover`` samples images from each entity and creates contact sheets.
   With an OpenAI-compatible vision API it asks for a constrained facet schema.
2. ``assign`` loads an existing Violas pickle cache and assigns every cached
   image to the discovered facets with CLIP text/image embeddings. No API call
   is needed for this step.
3. ``validate`` checks coverage and writes a compact manifest.

Only derived files are written. The original dataset and benchmark pickle are
never modified.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FACET_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
VALUE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_no}")
        rows.append(value)
    return rows


def safe_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return value or "entity"


def image_files(entity_dir: Path) -> list[Path]:
    return sorted(
        (p for p in entity_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def entities(
    dataset_root: Path,
    max_entities: int | None,
    selected: list[str] | None,
    excluded: list[str] | None = None,
) -> list[tuple[str, list[Path]]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    wanted = set(selected or [])
    blocked = {"BACKGROUND_Google", *(excluded or [])}
    found: list[tuple[str, list[Path]]] = []
    for entity_dir in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not entity_dir.is_dir() or entity_dir.name in blocked:
            continue
        if wanted and entity_dir.name not in wanted:
            continue
        files = image_files(entity_dir)
        if files:
            found.append((entity_dir.name, files))
    if max_entities is not None:
        found = found[: max(0, max_entities)]
    return found


def object_id(entity: str, image_path: Path, dataset_root: Path) -> str:
    """Stable ID based on dataset-relative path, never on an absolute path."""
    return f"{entity}/{image_path.name}"


def relative_path(image_path: Path, dataset_root: Path) -> str:
    try:
        return image_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return image_path.name


def sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    rng = random.Random(seed)
    # Sorted random sampling is deterministic and keeps contact sheets readable.
    return sorted(rng.sample(paths, count), key=lambda p: p.name)


def diverse_sample_paths(
    paths: list[Path],
    vector_by_relative_path: dict[str, Any],
    dataset_root: Path,
    count: int,
    seed: int,
) -> list[Path]:
    """Select a deterministic visual coreset with farthest-point sampling.

    The existing Violas cache already contains CLIP image vectors. We use them
    only to cover the entity's visual space; no image is sent to the API during
    this step. If vectors are missing, fall back to the regular deterministic
    random sample rather than silently producing a partial gallery.
    """
    if len(paths) <= count:
        return paths
    import numpy as np

    keys = [relative_path(path, dataset_root) for path in paths]
    available = [index for index, key in enumerate(keys) if key in vector_by_relative_path]
    if len(available) < max(4, min(count, len(paths)) // 2):
        return sample_paths(paths, count, seed)
    matrix = np.asarray([vector_by_relative_path[key] for key in keys], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    rng = random.Random(seed)
    first = rng.randrange(len(paths))
    selected = [first]
    min_distance = 1.0 - (matrix @ matrix[first])
    min_distance[first] = -1.0
    while len(selected) < min(count, len(paths)):
        next_index = int(np.argmax(min_distance))
        if next_index in selected:
            break
        selected.append(next_index)
        distance = 1.0 - (matrix @ matrix[next_index])
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1.0
    return sorted((paths[index] for index in selected), key=lambda p: p.name)


def make_contact_sheet(paths: list[Path], labels: list[str], output: Path, tile_size: int = 180, columns: int = 8) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = (len(paths) + columns - 1) // columns
    label_height = 24
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for index, (path, label) in enumerate(zip(paths, labels)):
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_size - 6, tile_size - 6))
                left = x + (tile_size - image.width) // 2
                top = y + (tile_size - image.height) // 2
                sheet.paste(image, (left, top))
        except Exception as exc:
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline="red")
            draw.text((x + 4, y + 4), f"read error: {exc}", fill="red", font=font)
        draw.text((x + 4, y + tile_size + 3), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=90)


def image_data_url(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def discovery_prompt(entity: str, labels: list[str]) -> str:
    joined = ", ".join(labels)
    return f"""You are defining diversity facets for one visual entity.

The entity is: {entity!r}.
The attached contact sheet contains images identified by these IDs: {joined}.

Propose 2 to 5 recurring, visually distinguishable semantic facet dimensions
that are actually supported by this gallery and useful for structured search.
For every facet, define 2 to 6 mutually exclusive values. A facet is a key
(for example ``flight_state``); its values are the possible labels within that
dimension (for example ``airborne`` and ``ground``). Do not return dimensions
as competing labels. Do not invent facts that cannot be seen. Omit a facet if
the gallery shows only one value or if its values cannot be visually separated.
Every value must cite one or more supporting image IDs from the gallery.

Return JSON only, with exactly this shape:
{{
  "facets": [
    {{
      "id": "lowercase_snake_case_key",
      "name": "short dimension name",
      "description": "what this dimension measures",
      "values": [
        {{
          "id": "lowercase_snake_case_value",
          "name": "short value name",
          "description": "visual definition of this value",
          "evidence_image_ids": ["{labels[0] if labels else 'image_001'}"]
        }}
      ]
    }}
  ]
}}

Facet and value IDs must be unique lowercase snake_case strings within their
scope. Evidence IDs must be copied from the contact sheet IDs. Return JSON only."""


def validate_schema(entity: str, value: Any, valid_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("facets"), list):
        raise ValueError("response must be an object with a facets array")
    facets = value["facets"]
    if not 1 <= len(facets) <= 6:
        raise ValueError("facets must contain between 1 and 6 entries")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for facet in facets:
        if not isinstance(facet, dict):
            raise ValueError("each facet must be an object")
        facet_id = str(facet.get("id", "")).strip()
        name = str(facet.get("name", "")).strip()
        description = str(facet.get("description", "")).strip()
        values = facet.get("values", [])
        if not FACET_ID_RE.fullmatch(facet_id):
            raise ValueError(f"invalid facet id: {facet_id!r}")
        if facet_id in seen:
            raise ValueError(f"duplicate facet id: {facet_id}")
        if not name or not description:
            raise ValueError(f"facet {facet_id} needs name and description")
        if not isinstance(values, list) or not 2 <= len(values) <= 8:
            raise ValueError(f"facet {facet_id} must contain between 2 and 8 values")
        value_seen: set[str] = set()
        normalized_values: list[dict[str, Any]] = []
        for option in values:
            if not isinstance(option, dict):
                raise ValueError(f"facet {facet_id} contains a non-object value")
            value_id = str(option.get("id", "")).strip()
            value_name = str(option.get("name", "")).strip()
            value_description = str(option.get("description", "")).strip()
            evidence = option.get("evidence_image_ids", [])
            if not VALUE_ID_RE.fullmatch(value_id):
                raise ValueError(f"facet {facet_id} has invalid value id: {value_id!r}")
            if value_id in value_seen:
                raise ValueError(f"facet {facet_id} has duplicate value id: {value_id}")
            if not value_name or not value_description:
                raise ValueError(f"facet {facet_id} value {value_id} needs name and description")
            if not isinstance(evidence, list) or not evidence or any(str(x) not in valid_ids for x in evidence):
                raise ValueError(f"facet {facet_id} value {value_id} has invalid evidence_image_ids")
            value_seen.add(value_id)
            normalized_values.append({
                "id": value_id,
                "name": value_name,
                "description": value_description,
                "evidence_image_ids": [str(x) for x in evidence],
            })
        seen.add(facet_id)
        normalized.append({
            "id": facet_id,
            "name": name,
            "description": description,
            "values": normalized_values,
        })
    return {"entity": entity, "facets": normalized, "schema_version": "diversity-kv-v2"}


class VisionApiClient:
    """Small OpenAI-compatible client using only Python's standard library."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120, retries: int = 1):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def discover(
        self, entity: str, paths: list[Path], prompt: str, *, request_options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in paths:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path), "detail": "low"}})
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        if request_options:
            body.update(request_options)
        endpoint = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                payload = json.loads(raw)
                text = payload["choices"][0]["message"]["content"]
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                text = str(text).strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
                return json.loads(text), {"response": payload, "attempt": attempt}
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    detail = ""
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
            except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"vision API failed after {self.retries + 1} attempts: {last_error}") from last_error


def resolve_api_key(env_name: str) -> str:
    """Read an explicitly supplied environment credential."""
    return os.environ.get(env_name, "").strip()


def load_cache_vectors(cache_path: Path, dataset_root: Path) -> dict[str, dict[str, Any]]:
    import pickle

    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    by_entity: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in iter_cache_objects(cache, dataset_root):
        relative = str(item["relative_path"])
        # Older cache descriptions store paths relative to the project root,
        # while the dataset discovery code uses paths relative to dataset_root.
        # Normalize the former ``image_0001.jpg`` form to ``entity/image``.
        if "/" not in relative and "\\" not in relative:
            relative = f"{item['entity']}/{relative}"
        by_entity[item["entity"]][relative] = item["vector"]
    return by_entity


def load_schemas(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["entity"]): row for row in read_jsonl(path)}


def update_manifest(output_dir: Path, values: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values)
    json_dump(path, current)


def command_discover(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root).resolve()
    out = Path(args.output_dir).resolve()
    sheet_dir = out / "contact_sheets"
    raw_dir = out / "raw_api"
    schemas_path = out / "schemas.jsonl"
    errors_path = out / "errors.jsonl"
    selected = args.entity or None
    groups = entities(root, args.max_entities, selected, args.exclude_entity)
    if not groups:
        raise RuntimeError("No image entities found")
    existing = load_schemas(schemas_path)
    vector_cache: dict[str, dict[str, Any]] = {}
    if args.sample_strategy == "diverse":
        if not args.cache:
            raise RuntimeError("--sample-strategy diverse requires --cache")
        vector_cache = load_cache_vectors(Path(args.cache), root)
    api_key = resolve_api_key(args.api_key_env)
    client = None
    if not args.prepare_only:
        if not api_key:
            raise RuntimeError(f"Missing API key. Set environment variable {args.api_key_env}, or use --prepare-only.")
        client = VisionApiClient(api_key, args.api_base, args.model, args.timeout, args.retries)
    prepared: list[dict[str, Any]] = []
    jobs: list[tuple[str, list[str], Path, dict[str, Any]]] = []
    for entity, paths in groups:
        if entity in existing and not args.force:
            print(f"[skip] {entity}: schema already exists")
            continue
        if args.sample_strategy == "diverse":
            sample = diverse_sample_paths(
                paths,
                vector_cache.get(entity, {}),
                root,
                args.sample_per_entity,
                args.seed,
            )
        else:
            sample = sample_paths(paths, args.sample_per_entity, args.seed)
        labels = [f"image_{i:03d}" for i in range(len(sample))]
        sheet_path = sheet_dir / f"{safe_id(entity)}.jpg"
        make_contact_sheet(sample, labels, sheet_path, args.tile_size, args.columns)
        row = {
            "dataset": args.dataset,
            "entity": entity,
            "sample_image_ids": labels,
            "sample_relative_paths": [relative_path(p, root) for p in sample],
            "contact_sheet": str(sheet_path.relative_to(out)).replace("\\", "/"),
            "model": args.model,
            "prompt_version": "facet-values-discovery-v2",
            "sample_strategy": args.sample_strategy,
        }
        prepared.append(row)
        if args.prepare_only:
            print(f"[prepared] {entity}: {sheet_path}")
            continue
        jobs.append((entity, labels, sheet_path, row))

    if jobs:
        assert client is not None
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def discover_one(job: tuple[str, list[str], Path, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
            entity, labels, sheet_path, row = job
            base_prompt = discovery_prompt(entity, labels)
            prompt = base_prompt
            for schema_attempt in range(args.schema_retries + 1):
                # Send one labeled gallery instead of all raw images. This
                # keeps the request small while covering the entity globally.
                parsed, raw = client.discover(entity, [sheet_path], prompt)
                try:
                    schema = validate_schema(entity, parsed, set(labels))
                    schema.update({k: row[k] for k in ("dataset", "sample_image_ids", "sample_relative_paths", "contact_sheet", "model", "prompt_version")})
                    raw["schema_attempt"] = schema_attempt
                    return schema, raw
                except ValueError as exc:
                    if schema_attempt >= args.schema_retries:
                        raise
                    prompt = (
                        base_prompt
                        + "\n\nYour previous JSON failed validation: "
                        + str(exc)
                        + "\nReturn a corrected JSON object only."
                    )
            raise AssertionError("unreachable")

        print(f"Starting {len(jobs)} schema calls with workers={max(1, args.workers)}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_to_job = {pool.submit(discover_one, job): job for job in jobs}
            for future in as_completed(future_to_job):
                entity, _labels, _sheet_path, _row = future_to_job[future]
                try:
                    schema, raw = future.result()
                    append_jsonl(schemas_path, schema)
                    json_dump(raw_dir / f"{safe_id(entity)}.json", raw)
                    existing[entity] = schema
                    print(f"[ok] {entity}: {len(schema['facets'])} facets", flush=True)
                except Exception as exc:
                    append_jsonl(errors_path, {"stage": "discover", "entity": entity, "error": repr(exc)})
                    print(f"[error] {entity}: {exc}", file=sys.stderr, flush=True)
    json_dump(out / "requests.json", prepared)
    update_manifest(out, {
        "format": "violas-diversity-kv-v2",
        "dataset": args.dataset,
        "dataset_root": str(root),
        "schema_source_split": "train",
        "model": args.model,
        "prompt_version": "facet-values-discovery-v2",
        "sample_strategy": args.sample_strategy,
        "sample_cache": str(Path(args.cache).resolve()) if args.cache else None,
        "max_entities": args.max_entities,
        "selected_entities": [entity for entity, _ in groups],
        "excluded_entities": ["BACKGROUND_Google", *args.exclude_entity],
        # Keep this count meaningful when a later single-entity retry updates
        # the same output directory: it reflects all schemas currently on disk,
        # not only schemas created in this invocation.
        "schema_entities_completed": len(existing),
        "prepare_only": bool(args.prepare_only),
    })
    print(f"Saved derived data under: {out}")


def iter_cache_objects(cache: dict[str, Any], dataset_root: Path) -> Iterable[dict[str, Any]]:
    for split in ("train", "test"):
        cache_key = "train_data_by_key" if split == "train" else "test_data"
        for entity, data in cache.get(cache_key, {}).items():
            vectors = data.get("vectors", [])
            descriptions = data.get("descriptions", [])
            for index, vector in enumerate(vectors):
                desc = descriptions[index] if index < len(descriptions) else {}
                raw_path = desc.get("img_path")
                if not raw_path:
                    continue
                path = Path(str(raw_path))
                yield {
                    "object_id": object_id(str(entity), path, dataset_root),
                    "entity": str(entity),
                    "split": split,
                    "relative_path": relative_path(path, dataset_root),
                    "vector": vector,
                }


def command_assign(args: argparse.Namespace) -> None:
    import pickle
    import numpy as np

    root = Path(args.dataset_root).resolve()
    out = Path(args.output_dir).resolve()
    schemas = load_schemas(out / "schemas.jsonl")
    if not schemas:
        raise RuntimeError(f"No schemas found at {out / 'schemas.jsonl'}")
    with Path(args.cache).open("rb") as stream:
        cache = pickle.load(stream)
    value_vectors: dict[str, dict[str, np.ndarray]] = {}
    if args.assignment_method == "prototype":
        cached_vectors = load_cache_vectors(Path(args.cache), root)
        for entity, schema in schemas.items():
            value_vectors[entity] = {}
            evidence_paths = dict(zip(schema.get("sample_image_ids", []), schema.get("sample_relative_paths", [])))
            for facet in schema["facets"]:
                prototypes: list[np.ndarray] = []
                for option in facet["values"]:
                    evidence_vectors: list[np.ndarray] = []
                    for image_id in option.get("evidence_image_ids", []):
                        path = evidence_paths.get(image_id)
                        vector = cached_vectors.get(entity, {}).get(str(path)) if path else None
                        if vector is not None:
                            evidence_vectors.append(np.asarray(vector, dtype=np.float32).reshape(-1))
                    if not evidence_vectors:
                        raise RuntimeError(f"no cached evidence vectors for {entity}.{facet['id']}={option['id']}")
                    matrix = np.stack(evidence_vectors)
                    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
                    prototype = matrix.mean(axis=0)
                    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
                    prototypes.append(prototype)
                value_vectors[entity][facet["id"]] = np.stack(prototypes)
        assignment_model = "evidence-image-prototypes"
        assignment_source = "schema_evidence_vectors_from_existing_cache"
    else:
        try:
            import clip
            import torch
        except ImportError as exc:
            raise RuntimeError("clip_text assignment requires the existing CLIP dependency") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load(args.clip_model, device=device)
        model.eval()
        with torch.no_grad():
            for entity, schema in schemas.items():
                value_vectors[entity] = {}
                for facet in schema["facets"]:
                    prompts = [
                        f"a photo of {entity}; {facet['name']}: {option['name']}. {option['description']}"
                        for option in facet["values"]
                    ]
                    tokens = clip.tokenize(prompts).to(device)
                    values = model.encode_text(tokens).float().cpu().numpy()
                    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
                    value_vectors[entity][facet["id"]] = values
        assignment_model = args.clip_model
        assignment_source = "existing_cache_vectors_plus_clip_text_embeddings"
    rows: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    facet_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    uncertain_assignments = 0
    for item in iter_cache_objects(cache, root):
        entity = item["entity"]
        schema = schemas.get(entity)
        if schema is None:
            missing[entity] += 1
            continue
        vector = np.asarray(item["vector"], dtype=np.float32).reshape(-1)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        attributes: dict[str, Any] = {}
        tokens: list[str] = []
        object_uncertain = False
        for facet in schema["facets"]:
            facet_id = facet["id"]
            scores = value_vectors[entity][facet_id] @ vector
            order = np.argsort(-scores)
            top = int(order[0])
            second = int(order[1]) if len(order) > 1 else top
            options = facet["values"]
            value_id = options[top]["id"]
            confidence = float(scores[top])
            margin = float(scores[top] - scores[second])
            uncertain = margin < args.uncertain_margin
            attributes[facet_id] = {
                "value": value_id,
                "confidence": confidence,
                "margin": margin,
                "uncertain": uncertain,
                "alternatives": [
                    {"value": options[int(index)]["id"], "score": float(scores[int(index)])}
                    for index in order[: min(3, len(order))]
                ],
            }
            tokens.append(f"{facet_id}={value_id}")
            facet_counts[entity][facet_id][value_id] += 1
            uncertain_assignments += int(uncertain)
            object_uncertain = object_uncertain or uncertain
        row = {
            "object_id": item["object_id"],
            "dataset": args.dataset,
            "entity": entity,
            "split": item["split"],
            "relative_path": item["relative_path"],
            "attributes": attributes,
            "tokens": tokens,
            "uncertain": object_uncertain,
            "source": args.assignment_method,
            "schema_version": "diversity-kv-v2",
        }
        rows.append(row)
    assignments_path = out / "assignments.jsonl"
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    update_manifest(out, {
        "assignment_method": args.assignment_method,
        "assignment_model": assignment_model,
        "assignment_source": assignment_source,
        "assignment_count": len(rows),
        "missing_schema_objects": dict(missing),
        "facet_counts": {
            entity: {facet_id: dict(counts) for facet_id, counts in by_facet.items()}
            for entity, by_facet in facet_counts.items()
        },
        "uncertain_assignment_count": uncertain_assignments,
        "uncertain_object_count": sum(bool(row["uncertain"]) for row in rows),
        "source_cache": str(Path(args.cache).resolve()),
    })
    print(
        f"Assigned {len(rows)} cached objects across KV facets; "
        f"uncertain_assignments={uncertain_assignments}, "
        f"uncertain_objects={sum(bool(row['uncertain']) for row in rows)}"
    )


def command_validate(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    schemas = load_schemas(out / "schemas.jsonl")
    assignments = read_jsonl(out / "assignments.jsonl")
    errors: list[str] = []
    for entity, schema in schemas.items():
        try:
            validate_schema(entity, schema, set(schema.get("sample_image_ids", [])))
        except Exception as exc:
            errors.append(f"schema {entity}: {exc}")
    seen: set[str] = set()
    for row in assignments:
        object_key = str(row.get("object_id", ""))
        if not object_key or object_key in seen:
            errors.append(f"duplicate or empty object_id: {object_key!r}")
        seen.add(object_key)
        entity = row.get("entity")
        if entity not in schemas:
            errors.append(f"assignment has no schema: {entity!r}")
            continue
        attributes = row.get("attributes")
        if not isinstance(attributes, dict):
            errors.append(f"assignment {object_key!r} has no attributes object")
            continue
        schema = schemas[entity]
        expected = {facet["id"]: facet for facet in schema["facets"]}
        if set(attributes) != set(expected):
            errors.append(f"assignment {object_key!r} facet keys do not match schema")
            continue
        for facet_id, facet in expected.items():
            allowed = {option["id"] for option in facet["values"]}
            assigned = attributes[facet_id]
            if not isinstance(assigned, dict) or assigned.get("value") not in allowed:
                errors.append(f"assignment {object_key!r} has invalid value for {facet_id}")
    report = {"schema_count": len(schemas), "assignment_count": len(assignments), "errors": errors, "valid": not errors}
    json_dump(out / "validation.json", report)
    if errors:
        raise RuntimeError("Validation failed; see validation.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="caltech101")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-entities", type=int, default=3, help="default 3 for the first smoke experiment")
    parser.add_argument("--entity", action="append", help="specific entity name; can be repeated")
    parser.add_argument("--exclude-entity", action="append", default=[], help="entity name to skip; can be repeated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="make contact sheets and discover facet schemas")
    add_common(discover)
    discover.add_argument("--sample-per-entity", type=int, default=32)
    discover.add_argument("--sample-strategy", choices=("random", "diverse"), default="random")
    discover.add_argument("--cache", help="existing Violas pickle cache; required for --sample-strategy diverse")
    discover.add_argument("--seed", type=int, default=42)
    discover.add_argument("--tile-size", type=int, default=180)
    discover.add_argument("--columns", type=int, default=8)
    discover.add_argument("--prepare-only", action="store_true", help="do not call the API; only create sheets and requests.json")
    discover.add_argument("--force", action="store_true")
    discover.add_argument("--api-key-env", default="VIOLAS_VLM_API_KEY")
    discover.add_argument("--api-base", default="https://api.openai.com/v1")
    discover.add_argument("--model", default="gpt-4o-mini")
    discover.add_argument("--timeout", type=int, default=120)
    discover.add_argument("--retries", type=int, default=1)
    discover.add_argument("--schema-retries", type=int, default=1, help="retry a model response that fails KV schema validation")
    discover.add_argument("--workers", type=int, default=1, help="parallel schema API calls")
    discover.set_defaults(func=command_discover)
    assign = sub.add_parser("assign", help="assign cached images to every facet value")
    add_common(assign)
    assign.add_argument("--cache", required=True)
    assign.add_argument("--clip-model", default="ViT-B/32")
    assign.add_argument("--assignment-method", choices=("prototype", "clip_text"), default="prototype")
    assign.add_argument("--uncertain-margin", type=float, default=0.03)
    assign.set_defaults(func=command_assign)
    validate = sub.add_parser("validate", help="validate derived schemas and assignments")
    validate.add_argument("--output-dir", required=True)
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
