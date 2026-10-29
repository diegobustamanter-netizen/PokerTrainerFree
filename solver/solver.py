"""
Solver Nash push/fold para Sit & Go de 3 jugadores (y heads-up).

Supuestos:
  - Ciegas 0.5 / 1 bb, sin ante, stacks iguales de S bb.
  - Solo hay dos acciones: all-in o fold (y call/fold frente a un all-in).
  - Se maximiza chipEV. En formatos winner-takes-all (Spin & Go, Twister,
    Expresso...) el chipEV coincide con el $EV, así que es el criterio correcto.

Entrada: archivos .bin generados por sampler.c (tablas 169x169x169).
Salida:  data.json con el ΔEV (en centésimas de bb) de la acción agresiva
         frente a fold, por spot, stack y clase de mano.

Uso: python3 solver.py s1.bin s2.bin ... -o data.json
"""
import argparse
import json
import time

import numpy as np

DT = np.dtype([('n', '<u4'), ('ebs', '<u4'), ('ebg', '<u4'),
               ('esg', '<u4'), ('tb', '<u4'), ('ts', '<u4')])
H = 169


def load(files):
    acc = None
    for f in files:
        a = np.fromfile(f, dtype=DT)
        if acc is None:
            acc = {k: a[k].astype(np.float64) for k in DT.names}
        else:
            for k in DT.names:
                acc[k] += a[k]
    shp = (H, H, H)
    N = acc['n'].reshape(shp)
    EBS = (acc['ebs'] / 2).reshape(shp)   # equity BTN vs SB (heads-up)
    EBG = (acc['ebg'] / 2).reshape(shp)   # equity BTN vs BB
    ESG = (acc['esg'] / 2).reshape(shp)   # equity SB vs BB
    TB = (acc['tb'] / 6).reshape(shp)     # cuota 3-way del BTN
    TS = (acc['ts'] / 6).reshape(shp)     # cuota 3-way del SB
    TG = N - TB - TS                      # cuota 3-way del BB
    return N, EBS, EBG, ESG, TB, TS, TG


class Tables:
    """Las mismas tablas en tres órdenes de ejes para multiplicar rápido."""

    def __init__(self, N, EBS, EBG, ESG, TB, TS, TG):
        f = lambda a: np.ascontiguousarray(a, dtype=np.float32).reshape(H, -1)
        # eje 0 = mano del BTN (i), luego (j, k)
        self.iN, self.iEBS, self.iEBG, self.iTB = f(N), f(EBS), f(EBG), f(TB)
        # eje 0 = mano del SB (j), luego (i, k)
        t = lambda a: a.transpose(1, 0, 2)
        self.jN, self.jEBS, self.jTS, self.jESG = f(t(N)), f(t(EBS)), f(t(TS)), f(t(ESG))
        # eje 0 = mano del BB (k), luego (i, j)
        t = lambda a: a.transpose(2, 0, 1)
        self.kN, self.kEBG, self.kESG, self.kTG = f(t(N)), f(t(EBG)), f(t(ESG)), f(t(TG))
        self.iDen = self.iN.sum(1)
        # heads-up: se marginaliza la mano del tercer jugador
        self.huN = N.sum(0)
        self.huE = ESG.sum(0)


def mv(A, x, y):
    return A @ np.outer(x, y).astype(np.float32).ravel()


def deltas3(T, S, st):
    """ΔEV (sumas) de la acción agresiva menos fold, y denominadores."""
    pB, cS, cG1, cG2, pS, cG3 = (st[k] for k in ('btn_push', 'sb_call', 'bb_call_btn',
                                                  'bb_overcall', 'sb_push', 'bb_call_sb'))
    one = np.ones(H)
    d, den = {}, {}
    # BTN: all-in vs fold
    a, b = (1 - cS), cS
    d['btn_push'] = (mv(T.iN, a, 1.5 * (1 - cG1) - S * cG1) + mv(T.iN, b, -S * one)
                     + mv(T.iEBG, a, (2 * S + .5) * cG1)
                     + mv(T.iEBS, b, (2 * S + 1) * (1 - cG2))
                     + mv(T.iTB, b, 3 * S * cG2))
    den['btn_push'] = T.iDen
    # SB frente al all-in del BTN: call vs fold
    u0 = mv(T.jN, pB, one)
    d['sb_call'] = (mv(T.jN, pB, (2 * S + 1) * (1 - cG2)) + (0.5 - S) * u0
                    - mv(T.jEBS, pB, (2 * S + 1) * (1 - cG2))
                    + mv(T.jTS, pB, 3 * S * cG2))
    den['sb_call'] = u0
    # BB frente al all-in del BTN, SB foldea
    w = mv(T.kN, pB, 1 - cS)
    d['bb_call_btn'] = (S + 1.5) * w - (2 * S + .5) * mv(T.kEBG, pB, 1 - cS)
    den['bb_call_btn'] = w
    # BB frente a all-in del BTN y call del SB (overcall)
    w = mv(T.kN, pB, cS)
    d['bb_overcall'] = 3 * S * mv(T.kTG, pB, cS) - (S - 1) * w
    den['bb_overcall'] = w
    # SB tras fold del BTN: all-in vs fold
    q = 1 - pB
    u0 = mv(T.jN, q, one)
    d['sb_push'] = (mv(T.jN, q, (1 - cG3) - S * cG3) + 0.5 * u0
                    + 2 * S * mv(T.jESG, q, cG3))
    den['sb_push'] = u0
    # BB frente al all-in del SB tras fold del BTN
    w = mv(T.kN, q, pS)
    d['bb_call_sb'] = (S + 1) * w - 2 * S * mv(T.kESG, q, pS)
    den['bb_call_sb'] = w
    return d, den


def solve3(T, S, init=None, iters=800, w0=20):
    keys = ('btn_push', 'sb_call', 'bb_call_btn', 'bb_overcall', 'sb_push', 'bb_call_sb')
    st = {k: (init[k].copy() if init else np.full(H, 0.5)) for k in keys}
    start = w0 if init else 0
    for t in range(iters):
        d, _ = deltas3(T, S, st)
        lr = 1.0 / (start + t + 2)
        for k in keys:
            br = (d[k] > 0).astype(float)
            st[k] += (br - st[k]) * lr
    d, den = deltas3(T, S, st)
    tot = T.iN.sum()
    expl = {k: float((np.maximum(d[k], 0) - st[k] * d[k]).sum() / tot) for k in keys}
    ev = {k: d[k] / np.maximum(den[k], 1e-9) for k in keys}
    return st, ev, expl


def solve_hu(T, S, iters=3000):
    N2, E2 = T.huN, T.huE
    p, c = np.full(H, 0.5), np.full(H, 0.5)
    for t in range(iters):
        dp = N2 @ (1.5 - c * (S + 1)) + 2 * S * (E2 @ c)
        dc = (S + 1) * (p @ N2) - 2 * S * (p @ E2)
        lr = 1.0 / (t + 2)
        p += ((dp > 0) - p) * lr
        c += ((dc > 0) - c) * lr
    dp = N2 @ (1.5 - c * (S + 1)) + 2 * S * (E2 @ c)
    dc = (S + 1) * (p @ N2) - 2 * S * (p @ E2)
    return {'hu_sb_push': dp / N2.sum(1), 'hu_bb_call': dc / np.maximum(p @ N2, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bins', nargs='+')
    ap.add_argument('-o', '--out', default='data.json')
    ap.add_argument('--smin', type=int, default=2)
    ap.add_argument('--smax', type=int, default=15)
    ap.add_argument('--iters', type=int, default=800)
    args = ap.parse_args()

    T = Tables(*load(args.bins))
    stacks = list(range(args.smin, args.smax + 1))
    out = {'stacks': stacks, 'ev': {}}
    prev = None
    for S in stacks:
        t0 = time.time()
        st, ev, expl = solve3(T, S, prev, iters=args.iters)
        prev = st
        ev.update(solve_hu(T, S))
        for k, v in ev.items():
            out['ev'].setdefault(k, []).append([int(round(x * 100)) for x in v])
        print(f'S={S:2d}bb  {time.time() - t0:5.1f}s  explotabilidad (bb/mano): '
              + ' '.join(f'{k}={v:.4f}' for k, v in expl.items()), flush=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
