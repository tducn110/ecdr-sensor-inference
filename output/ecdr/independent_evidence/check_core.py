import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path('/home/pro/Downloads/research')
sys.path.insert(0, str(ROOT))
from ecdr.model import OperationalSequentialNumpyCNN
from ecdr.incremental import IncrementalEngine
from ecdr.intervals import Interval
from ecdr.bounds import (
    compute_diagonal_box_certificate, compute_wave_domain_certificate,
    interval_conv_relu_vectorized, interval_gap_canonical, interval_affine_head,
)

records = []
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'ecdr').glob('*.py')}


def check(name, fn):
    try:
        detail = fn()
        records.append({'name': name, 'status': 'PASS', 'detail': detail})
    except Exception as exc:
        records.append({'name': name, 'status': 'FAIL',
                        'error': type(exc).__name__ + ': ' + str(exc)})


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(fn):
    try:
        result = fn()
    except (ValueError, FloatingPointError, OverflowError):
        return {'rejected': True}
    if hasattr(result, 'is_certified') and not result.is_certified:
        return {'rejected': True, 'method': result.method}
    raise AssertionError('Invalid input was accepted')


def random_net(seed):
    rng = np.random.default_rng(seed)
    bhead = rng.normal(scale=.01, size=12)
    bhead[0] += 4.
    return OperationalSequentialNumpyCNN(
        rng.normal(scale=.05, size=(32, 6, 5)), rng.normal(scale=.01, size=32),
        rng.normal(scale=.05, size=(64, 32, 3)), rng.normal(scale=.01, size=64),
        rng.normal(scale=.05, size=(64, 64, 3)), rng.normal(scale=.01, size=64),
        rng.normal(scale=.05, size=(12, 64)), bhead)


def sparse_net(multiplier=1., tie=False, overflow=False, negative_overflow=False):
    w1 = np.zeros((32, 6, 5)); w2 = np.zeros((64, 32, 3))
    w3 = np.zeros((64, 64, 3)); wh = np.zeros((12, 64))
    w1[0, 0, 0] = (-1e308 if negative_overflow else 1e308) if overflow else multiplier
    w2[0, 0, 0] = w3[0, 0, 0] = wh[0, 0] = 0. if tie else 1.
    return OperationalSequentialNumpyCNN(
        w1, np.zeros(32), w2, np.zeros(64), w3, np.zeros(64), wh, np.zeros(12))


def exact_conv_box(seed):
    rng = np.random.default_rng(seed)
    lo = rng.normal(size=(2, 8)); hi = lo + rng.uniform(0., .1, size=lo.shape)
    w = rng.normal(size=(3, 2, 3)); b = rng.normal(size=3)
    pre, post = interval_conv_relu_vectorized(Interval(lo, hi), w, b)
    count = 0
    for o in range(3):
        for t in range(6):
            exact_lo = Fraction(float(b[o])); exact_hi = exact_lo
            for c in range(2):
                for k in range(3):
                    ww = Fraction(float(w[o, c, k]))
                    a = Fraction(float(lo[c, t+k])); z = Fraction(float(hi[c, t+k]))
                    exact_lo += ww * (a if ww >= 0 else z)
                    exact_hi += ww * (z if ww >= 0 else a)
            require(Fraction(float(pre.lo[o, t])) <= exact_lo,
                    'Preactivation lower does not enclose exact Fraction lower')
            require(Fraction(float(pre.hi[o, t])) >= exact_hi,
                    'Preactivation upper does not enclose exact Fraction upper')
            require(Fraction(float(post.lo[o, t])) <= max(0, exact_lo),
                    'ReLU lower does not enclose exact Fraction lower')
            require(Fraction(float(post.hi[o, t])) >= max(0, exact_hi),
                    'ReLU upper does not enclose exact Fraction upper')
            count += 4
    return {'exact_endpoint_inequalities': count}


def full_point_enclosure(seed):
    net = random_net(seed); rng = np.random.default_rng(seed + 900)
    x = rng.normal(size=(6, 128)); out = net.forward(x)
    xx = Interval.from_point(x)
    for w, b in [(net.w1, net.b1), (net.w2, net.b2), (net.w3, net.b3)]:
        _, xx = interval_conv_relu_vectorized(xx, w, b)
    ll = interval_affine_head(interval_gap_canonical(xx), net.w_head, net.b_head)
    require(bool(np.all(ll.lo <= out['logits']) and np.all(out['logits'] <= ll.hi)),
            'Canonical executable logits escaped point interval')
    return {'logits_checked': len(out['logits'])}


def sampled_box_bound(seed, wave):
    net = random_net(seed); rng = np.random.default_rng(seed + 1900)
    anchor = rng.normal(size=(6, 128)); radius = .005
    center = rng.normal(size=(6, 64)) if wave else anchor
    if wave:
        box = Interval.from_box(center, radius)
        result = compute_wave_domain_certificate(net, anchor, box, shift=64, target_class=0)
    else:
        result = compute_diagonal_box_certificate(net, anchor, radius, target_class=0)
    require(result.is_certified, 'Strong-bias fixture failed to produce direct certificate')
    comparisons = 0
    for _ in range(4):
        sample = center + rng.uniform(-radius, radius, size=center.shape)
        xx = np.concatenate([anchor[:, 64:], sample], axis=1) if wave else sample
        logits = net.forward(xx)['logits']
        for j in range(1, 12):
            margin = float(logits[0] - logits[j])
            require(result.direct_lower_margins[j] <= margin,
                    f'Lower margin above actual program margin for competitor {j}')
            comparisons += 1
    return {'queries': 4, 'margin_comparisons': comparisons}


def multishift(shift):
    net = random_net(135); rng = np.random.default_rng(42 + shift)
    x = rng.normal(size=(6, 128)); engine = IncrementalEngine(net, shift=shift)
    engine.forward_full(x); compared = 0
    for i in range(4):
        x = np.concatenate([x[:, shift:], rng.normal(size=(6, shift))], axis=1)
        inc = engine.forward_incremental(x, step_idx=i+1); full = net.forward(x)
        for key in ('h1', 'h2', 'h3', 'm', 'logits'):
            require(np.array_equal(inc[key], full[key]),
                    f'Bitwise mismatch shift={shift}, step={i+1}, key={key}')
            compared += inc[key].size
    return {'steps': 4, 'array_elements_compared': compared}


def tiny_overlap():
    net = sparse_net(); engine = IncrementalEngine(net); x = np.ones((6, 128))
    engine.forward_full(x); y = x.copy(); y[0, 0] += 5e-13
    inc = engine.forward_incremental(y, step_idx=1); full = net.forward(y)
    require(np.array_equal(inc['h3'], full['h3']), 'Inexact overlap reused old features')
    return {'perturbation': 5e-13}


def interval_view_alias():
    parent = np.arange(4., dtype=np.float64); original = parent[:2].copy()
    box = Interval(parent[:2], parent[:2] + 2.)
    try:
        parent[0] = 100.
    except ValueError:
        pass
    require(np.array_equal(box.lo, original), 'Caller base mutation changed frozen bounds')
    return {'detached_or_immutable': True}


def model_parameter_alias():
    params = [np.ones((32, 6, 5)), np.zeros(32), np.ones((64, 32, 3)), np.zeros(64),
              np.ones((64, 64, 3)), np.zeros(64), np.ones((12, 64)), np.zeros(12)]
    net = OperationalSequentialNumpyCNN(*params)
    before = net.w1.copy(); params[0][:] = 2.
    require(np.array_equal(net.w1, before), 'Caller array alias changed model')
    try:
        net.w1[0, 0, 0] = 2.
    except ValueError:
        return {'detached_and_readonly': True}
    raise AssertionError('Weights permit in-place mutation')


def model_change():
    first = sparse_net(); second = sparse_net(2.)
    engine = IncrementalEngine(first); x = np.ones((6, 128)); engine.forward_full(x)
    engine.net = second
    inc = engine.forward_incremental(x, step_idx=1); full = second.forward(x)
    require(np.array_equal(inc['h3'], full['h3']), 'Changed model reused stale feature state')
    require(np.array_equal(inc['logits'], full['logits']), 'Changed model reused stale logits')
    return {'old_id': first.model_id, 'new_id': second.model_id}


def result_alias():
    net = sparse_net(); engine = IncrementalEngine(net); x = np.ones((6, 128))
    result = engine.forward_full(x)
    try:
        result['h3'][:] = 0.
    except ValueError:
        pass
    inc = engine.forward_incremental(x, step_idx=1); full = net.forward(x)
    require(np.array_equal(inc['h3'], full['h3']), 'Returned result mutation corrupted cache')
    return {'cache_isolated_from_returned_results': True}


def negative_threshold(wave):
    net = sparse_net(tie=True); x = np.ones((6, 128))
    if wave:
        fn = lambda: compute_wave_domain_certificate(net, x, Interval.from_point(x[:, 64:]),
                                                      target_class=1, margin_threshold=-2.)
    else:
        fn = lambda: compute_diagonal_box_certificate(net, x, 0., target_class=1,
                                                      margin_threshold=-2.)
    return rejected(fn)


def incremental_overflow():
    net = sparse_net(overflow=True); engine = IncrementalEngine(net)
    x = np.zeros((6, 128)); engine.forward_full(x)
    y = np.concatenate([x[:, 64:], np.full((6, 64), 1e308)], axis=1)
    return rejected(lambda: engine.forward_incremental(y, step_idx=1))


def reject_malformed_model():
    return rejected(lambda: OperationalSequentialNumpyCNN(
        np.zeros((32, 6, 3)), np.zeros(32),
        np.zeros((64, 32, 3)), np.zeros(64),
        np.zeros((64, 64, 3)), np.zeros(64),
        np.zeros((12, 64)), np.zeros(12)))


def diagonal_rounding_fixture():
    w1 = np.zeros((32, 6, 5)); w1[0, 0, 0] = 1e100
    w2 = np.zeros((64, 32, 3)); w2[0, 0, 0] = 1.
    w3 = np.zeros((64, 64, 3)); w3[0, 0, 0] = 1.
    wh = np.zeros((12, 64)); wh[1, 0] = 1e3
    bh = np.zeros(12); bh[0] = 1.
    net = OperationalSequentialNumpyCNN(
        w1, np.zeros(32), w2, np.zeros(64), w3, np.zeros(64), wh, bh)
    anchor = np.full((6, 128), -1.)
    query = np.zeros((6, 128)); query[0, :] = 1e-100
    return net, anchor, query


def diagonal_endpoint_membership():
    from ecdr.membership import check_diagonal_box_membership
    net, anchor, query = diagonal_rounding_fixture()
    box = Interval.from_box(anchor, 1.)
    require(not np.all(query <= box.hi), 'Fixture must lie outside actual certified endpoints')
    cert = compute_diagonal_box_certificate(net, anchor, 1., target_class=0)
    require(cert.is_certified, 'Fixture certificate unexpectedly inconclusive')
    actual = net.forward(query)['pred_class']
    require(actual == 1, 'Fixture reference must switch from class0 to class1')
    member = check_diagonal_box_membership(query, anchor, 1.)
    require(not member.is_member,
            'Rounded distance admitted outside-endpoint query with different class')
    return {'actual_class': actual, 'certified_target': 0,
            'query_coordinate': float(query[0, 0]), 'certified_upper': float(box.hi[0, 0])}


def diagonal_rounding_integration():
    import pandas as pd
    from ecdr.preprocessing import PreprocessingContract
    from ecdr.replay import StreamingReplayEngine
    net, anchor, query = diagonal_rounding_fixture()
    prep = PreprocessingContract.create(np.zeros(6), np.ones(6))
    replay = StreamingReplayEngine(net, prep, domain_type='diagonal_box', budget_norm=1.,
                                   use_incremental_reference=False, shadow_reference=True)
    manifest = pd.DataFrame({'exp': [1, 1], 'win_idx': [0, 1], 'start': [0, 64],
                             'end': [128, 192], 'subject': [1, 1], 'split': ['val', 'val'],
                             'label_policy': ['pure', 'pure'], 'label_majority': [1, 2]})
    replay.run_stream(np.stack([anchor, query]), manifest)
    record = replay.instrumentor.records[1]
    actual = net.forward(query)['pred_class']
    require(record.target_class == actual,
            f'Replay returned {record.target_class} for reference {actual}; hit={record.is_hit}')
    require(not record.is_hit, 'Outside-domain adversarial query was reused')
    return {'returned_class': record.target_class, 'reference_class': actual,
            'was_reused': record.is_hit}


started = time.perf_counter()
with np.errstate(all='ignore'):
    for seed in (11, 22, 33):
        check(f'exact_fraction_conv_box_{seed}', lambda s=seed: exact_conv_box(s))
        check(f'point_operational_enclosure_{seed}', lambda s=seed: full_point_enclosure(s))
        check(f'diagonal_sampled_margins_{seed}', lambda s=seed: sampled_box_bound(s, False))
        check(f'wave_sampled_margins_{seed}', lambda s=seed: sampled_box_bound(s, True))
    for shift in (1, 64, 120):
        check(f'bitwise_multishift_{shift}', lambda s=shift: multishift(s))
    check('tiny_overlap_fallback', tiny_overlap)
    for value, name in ((np.nan, 'nan'), (np.inf, 'inf')):
        xx = np.ones((6, 128)); xx[0, 0] = value
        check(f'reference_reject_{name}', lambda x=xx: rejected(lambda: sparse_net().forward(x)))
        def incremental_nonfinite(x=xx):
            eng = IncrementalEngine(sparse_net()); eng.forward_full(np.ones((6, 128)))
            return rejected(lambda: eng.forward_incremental(x, step_idx=1))
        check(f'incremental_reject_{name}', incremental_nonfinite)
    check('reference_reject_finite_overflow',
          lambda: rejected(lambda: sparse_net(overflow=True).forward(np.full((6, 128), 1e308))))
    check('reference_reject_negative_pre_relu_overflow',
          lambda: rejected(lambda: sparse_net(overflow=True, negative_overflow=True).forward(
              np.full((6, 128), 1e308))))
    check('incremental_reject_finite_overflow', incremental_overflow)
    check('reject_malformed_model_kernel', reject_malformed_model)
    check('diagonal_endpoint_rounding_membership', diagonal_endpoint_membership)
    check('diagonal_endpoint_rounding_integration', diagonal_rounding_integration)
    check('interval_reject_inverted', lambda: rejected(lambda: Interval(np.array([2.]), np.array([1.]))))
    check('interval_reject_nonfinite', lambda: rejected(lambda: Interval(np.array([np.nan]), np.array([np.inf]))))
    check('negative_threshold_diagonal', lambda: negative_threshold(False))
    check('negative_threshold_wave', lambda: negative_threshold(True))
    check('tie_not_certified', lambda: rejected(lambda: compute_diagonal_box_certificate(
        sparse_net(tie=True), np.ones((6, 128)), 0.)))
    check('frozen_interval_base_alias', interval_view_alias)
    check('frozen_model_parameter_alias', model_parameter_alias)
    check('model_change_invalidates_cache', model_change)
    check('returned_result_cache_alias', result_alias)

summary = {'cases': len(records), 'passed': sum(r['status']=='PASS' for r in records),
           'failed': sum(r['status']=='FAIL' for r in records),
           'elapsed_seconds': time.perf_counter()-started,
           'source_hashes': source_hashes, 'results': records}
output = Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/ecdr-k12-check/results.json')
output.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ('cases','passed','failed','elapsed_seconds')}))
for row in records:
    if row['status']=='FAIL': print(json.dumps(row))
