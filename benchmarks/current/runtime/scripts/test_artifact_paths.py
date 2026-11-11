"""Check path portability without datasets or VDB clients."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location('artifact_paths', Path(__file__).with_name('artifact_paths.py'))
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)
previous = os.environ.pop('VIOLAS_PATH_MAP_FILE', None)
try:
    record = {'source_cache': 'cache/example.pkl',
              'module_path': ['@repo/benchmarks/news20'], 'score': .75,
              'id': 'entity/item', 'img_path': 'dataset/images/a b.jpg'}
    plain = paths.rewrite(record)
    assert Path(plain['module_path'][0]).is_dir()
    assert plain['source_cache'] == record['source_cache']
    assert plain['score'] == .75 and plain['id'] == record['id']
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder)/'map.json'
        file.write_text(json.dumps({'cache': 'prepared/cache', 'cache/special': 'special'}), encoding='utf8')
        os.environ['VIOLAS_PATH_MAP_FILE'] = str(file)
        mapped = paths.rewrite(record)
        assert mapped['source_cache'] == str(Path('prepared/cache/example.pkl'))
        assert paths.rewrite('cache/special/item') == str(Path('special/item'))
        assert paths.rewrite('cached/item') == 'cached/item'
        assert record['source_cache'] == 'cache/example.pkl'
    assert paths.image_uri('dataset', 'a b.jpg') == (Path.cwd()/'dataset/a b.jpg').as_uri()
    print('PASS portable paths, repository modules, prefix matching, unchanged scores and image URIs')
finally:
    os.environ.pop('VIOLAS_PATH_MAP_FILE', None)
    if previous is not None:
        os.environ['VIOLAS_PATH_MAP_FILE'] = previous
