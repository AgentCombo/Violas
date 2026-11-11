"""Compare frozen Yahoo conditional search and native postfilter baselines."""
import argparse
import cProfile
import pstats
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns

from run_final_violas_bench import ROOT, load, read, write, soft
from adaptive_hdmg import AdaptiveHDMG
from adaptive_hdmg_before_yahoo import AdaptiveHDMG as OriginalHDMG
from run_caltech_filter_order import fetch, select_fetched


def run(profile=False):
    previous, config, source, _, _ = load('yahoo')
    storage = read(ROOT/'maintenance_v6/yahoo/vg_hdmg.json')
    saved = soft.bench.load_pickle(Path(storage['full_snapshot_path']), config.module_path)
    corpus = soft.Corpus(saved['cache'], saved['directory'])
    engine = AdaptiveHDMG(corpus)
    original_engine = OriginalHDMG(corpus)
    frozen = read(ROOT/'yahoo/results.json')
    bindings = soft.bench.choose_queries(previous['rows'], source, set(corpus.ids))
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for row in frozen['rows']:
            engine.search(bindings[row['id']]['normalized'], row['resolved_plan'], budget=32, score_batch=4)
        profiler.disable()
        pstats.Stats(profiler).sort_stats('cumtime').print_stats(25)
        return
    qdrant, _ = soft.setup_qdrant(config, corpus)
    _, chroma = soft.backends.open_chroma(Path('cache/five_system/yahoo_chroma'), 'yahoo_chroma')
    clients = dict(qdrant=qdrant, chroma=chroma)
    names = ('vg_before', 'vg_hdmg', 'qdrant', 'chroma')
    rows = []
    try:
        for repeat in range(5):
            for number, row in enumerate(frozen['rows']):
                query = bindings[row['id']]['normalized']
                assert bindings[row['id']]['object_id'] == row['query_object']['object_id']
                plan = row['resolved_plan']
                methods = {}
                shift = (number + repeat) % len(names)
                for name in names[shift:] + names[:shift]:
                    start = perf_counter_ns()
                    if name in ('vg_before', 'vg_hdmg'):
                        search = original_engine if name == 'vg_before' else engine
                        ids, scores, stats = search.search(query, plan, budget=32, score_batch=4)
                    else:
                        fetched = fetch(name, clients[name], config, corpus, query, plan, pre=False)
                        ids, scores, stats = select_fetched(fetched, corpus, query, plan, pre=False)
                    ms = (perf_counter_ns() - start) / 1e6
                    assert all(corpus.eligible(corpus.locator[oid], plan) for oid in ids)
                    value = soft.core.objective(ids, scores, corpus.objects, plan['facet_dimension'], plan['lambda'], plan['category_preferences'])
                    ratio = value / row['oracle']['objective']
                    assert ratio <= 1 + 2e-6
                    methods[name] = dict(ids=ids, objective_ratio=ratio, ms=ms, stats=stats)
                assert methods['vg_before']['ids'] == methods['vg_hdmg']['ids']
                before_stats = methods['vg_before']['stats']
                after_stats = methods['vg_hdmg']['stats']
                assert before_stats['unique_member_scores'] == after_stats['unique_member_scores']
                assert set(before_stats['selected_groups']) == set(after_stats['selected_groups'])
                assert methods['vg_before']['objective_ratio'] == methods['vg_hdmg']['objective_ratio']
                rows.append(dict(id=row['id'], repeat=repeat, methods=methods))
        aggregate = {name: dict(objective_ratio=fmean(r['methods'][name]['objective_ratio'] for r in rows),
            latency=soft.bench.latency([r['methods'][name]['ms'] for r in rows])) for name in names}
        out = ROOT/'yahoo_search_optimized'
        out.mkdir(parents=True, exist_ok=True)
        write(out/'results.json', dict(rows=rows, aggregate=aggregate, config=dict(repeats=5,
            workload_sha256=frozen['config']['workload_sha256'], candidate_k=config.candidate_k,
            vg_groups=32, score_batch=4, active_warmup=False)))
        print(aggregate, flush=True)
    finally:
        qdrant.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', action='store_true')
    run(parser.parse_args().profile)
