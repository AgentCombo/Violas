"""Portable artifact paths: workspace-relative data and @repo/ module paths."""
import json
import os
from pathlib import Path, PureWindowsPath

REPO = Path(__file__).resolve().parents[4]


def rewrite(value):
    name = os.environ.get('VIOLAS_PATH_MAP_FILE')
    mapping = json.loads(Path(name).read_text(encoding='utf8')) if name else {}

    def visit(x):
        if isinstance(x, dict):
            return {k: visit(v) for k, v in x.items()}
        if isinstance(x, list):
            return [visit(v) for v in x]
        if isinstance(x, str):
            if x.startswith('@repo/'):
                return str(REPO / x[len('@repo/'):])
            for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
                prefix = old.rstrip('/')
                if x == prefix or x.startswith(prefix + '/'):
                    return str(Path(new) / x[len(prefix):].lstrip('/'))
        return x

    return visit(value)


def image_uri(root, object_id):
    """Resolve relative image roots locally; accept explicit Windows roots too."""
    if PureWindowsPath(root).is_absolute():
        return (PureWindowsPath(root) / object_id).as_uri()
    return (Path(root).expanduser() / object_id).resolve().as_uri()
