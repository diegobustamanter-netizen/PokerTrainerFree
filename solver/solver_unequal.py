"""
Solver Nash push/fold de 3 jugadores con stacks DESIGUALES.

Normalización (exacta en chipEV): las fichas de un jugador por encima del mayor
stack rival nunca se pueden ganar ni perder, así que cada stack se reduce a
min(propio, máximo de los otros dos). Tras normalizar, los dos stacks mayores
son iguales: toda mesa queda descrita por
    (posición del corto, stack corto s, stack grande B), con s < B,
más el caso de stacks iguales (s == B).

En el bote a 3 bandas hay side pot: el bote principal (3*s) lo disputan los tres
y el side pot 2*(B - s) solo los dos stacks grandes.

Uso:
  python3 solver_unequal.py s1.bin s2.bin --part 0 --parts 2 -o part0.npz
  python3 solver_unequal.py --merge part0.npz part1.npz -o unequal.json
"""
import argparse
import base64
import json
import time

import numpy as np

from solver import H, Tables, load, mv

KEYS = ('btn_push', 'sb_call', 'bb_call_btn', 'bb_overcall', 'sb_push', 'bb_call_sb')
POS = ('BTN', 'SB', 'BB')


def deltas3u(T, st, Sb, Ss, Sg):
    """ΔEV (acción agresiva menos fold) para stacks normalizados (Sb, Ss, Sg)."""
    pB, cS, cG1, cG2, pS, cG3 = (st[k] for k in KEYS)
    one = np.ones(H)
    m = min(Sb, Ss, Sg)
    M3 = 3 * m
    big = {'b': Sb > m + 1e-9, 's': Ss > m + 1e-9, 'g': Sg > m + 1e-9}
    e_bs, e_bg, e_sg = min(Sb, Ss), min(Sb, Sg), min(Ss, Sg)
    d, den = {}, {}

    # --- BTN: all-in vs fold (fold = 0) ---
    a, b = 1 - cS, cS
    x = (mv(T.iN, a, 1.5 * (1 - cG1) - e_bg * cG1) + mv(T.iEBG, a, (2 * e_bg + .5) * cG1)
         + mv(T.iN, b, -e_bs * (1 - cG2)) + mv(T.iEBS, b, (2 * e_bs + 1) * (1 - cG2))
         + mv(T.iTB, b, M3 * cG2) - Sb * mv(T.iN, b, cG2))
    if big['b']:
        side = 2 * (Sb - m)
        x = x + side * (mv(T.iEBS, b, cG2) if big['s'] else mv(T.iEBG, b, cG2))
    d['btn_push'], den['btn_push'] = x, T.iDen

    # --- SB frente al all-in del BTN: call vs fold (fold = -0.5) ---
    u0 = mv(T.jN, pB, one)
    x = (mv(T.jN, pB, (e_bs + 1) * (1 - cG2)) - mv(T.jEBS, pB, (2 * e_bs + 1) * (1 - cG2))
         + mv(T.jTS, pB, M3 * cG2) - Ss * mv(T.jN, pB, cG2) + 0.5 * u0)
    if big['s']:
        side = 2 * (Ss - m)
        if big['b']:   # el otro grande es el BTN: equity del SB = 1 - EBS
            x = x + side * (mv(T.jN, pB, cG2) - mv(T.jEBS, pB, cG2))
        else:          # el otro grande es la BB
            x = x + side * mv(T.jESG, pB, cG2)
    d['sb_call'], den['sb_call'] = x, u0

    # --- BB frente al all-in del BTN, SB foldea (fold = -1) ---
    w = mv(T.kN, pB, 1 - cS)
    d['bb_call_btn'] = (e_bg + 1.5) * w - (2 * e_bg + .5) * mv(T.kEBG, pB, 1 - cS)
    den['bb_call_btn'] = w

    # --- BB overcall: BTN all-in y SB paga (fold = -1) ---
    w = mv(T.kN, pB, cS)
    x = M3 * mv(T.kTG, pB, cS) - (Sg - 1) * w
    if big['g']:
        side = 2 * (Sg - m)
        if big['b']:   # el otro grande es el BTN: equity de la BB = 1 - EBG
            x = x + side * (w - mv(T.kEBG, pB, cS))
        else:          # el otro grande es la SB: equity de la BB = 1 - ESG
            x = x + side * (w - mv(T.kESG, pB, cS))
    d['bb_overcall'], den['bb_overcall'] = x, w

    # --- SB tras fold del BTN: all-in vs fold (fold = -0.5) ---
    q = 1 - pB
    u0 = mv(T.jN, q, one)
    d['sb_push'] = (mv(T.jN, q, (1 - cG3) - e_sg * cG3) + 0.5 * u0
                    + 2 * e_sg * mv(T.jESG, q, cG3))
    den['sb_push'] = u0

    # --- BB frente al all-in de la SB (fold = -1) ---
    w = mv(T.kN, q, pS)
    d['bb_call_sb'] = (e_sg + 1) * w - 2 * e_sg * mv(T.kESG, q, pS)
    den['bb_call_sb'] = w
    return d, den


def solve(T, stacks, init=None, iters=400, w0=20):
    st = {k: (init[k].copy() if init else np.full(H, 0.5)) for k in KEYS}
    start = w0 if init else 0
    for t in range(iters):
        d, _ = deltas3u(T, st, *stacks)
        lr = 1.0 / (start + t + 2)
        for k in KEYS:
            st[k] += ((d[k] > 0).astype(float) - st[k]) * lr
    d, den = deltas3u(T, st, *stacks)
    tot = T.iN.sum()
    expl = max(float((np.maximum(d[k], 0) - st[k] * d[k]).sum() / tot) for k in KEYS)
    ev = {k: d[k] / np.maximum(den[k], 1e-9) for k in KEYS}
    return st, ev, expl


def stacks_for(pos, s, B):
    v = [B, B, B]
    v[pos] = s
    return tuple(float(x) for x in v)


def quant(ev_bb):
    """EV en bb -> int8 con escala raíz (más precisión cerca de 0)."""
    c = ev_bb * 100.0
    q = np.sign(c) * np.round(np.sqrt(np.abs(c)) * 2.8)
    return np.clip(q, -127, 127).astype(np.int8)


def run(args):
    T = Tables(*load(args.bins))
    Bs = [B for B in range(args.smin, args.smax + 1) if (B - args.smin) % args.parts == args.part]
    out = {}
    for B in Bs:
        t0 = time.time()
        st_eq, ev, expl = solve(T, (B, B, B), iters=args.eq_iters)
        out[f'eq_{B}'] = np.stack([quant(ev[k]) for k in KEYS])
        print(f'B={B:2d} iguales           expl={expl:.5f}  {time.time() - t0:5.1f}s', flush=True)
        for pos in range(3):
            prev = st_eq
            for s in range(B - 1, args.smin - 1, -1):
                t0 = time.time()
                prev, ev, expl = solve(T, stacks_for(pos, s, B), init=prev, iters=args.iters)
                out[f'{pos}_{s}_{B}'] = np.stack([quant(ev[k]) for k in KEYS])
                print(f'B={B:2d} {POS[pos]:3s} corto={s:2d}  expl={expl:.5f}  {time.time() - t0:5.1f}s', flush=True)
        np.savez(args.out, **out)
    np.savez(args.out, **out)


def merge(args):
    allv = {}
    for f in args.merge:
        z = np.load(f)
        allv.update({k: z[k] for k in z.files})
    smin, smax = args.smin, args.smax
    # Orden fijo: para cada B y cada s<=B, y cada posición del corto (s==B: una sola entrada)
    keys, blobs = [], []
    for B in range(smin, smax + 1):
        keys.append(f'eq_{B}')
        for pos in range(3):
            for s in range(smin, B):
                keys.append(f'{pos}_{s}_{B}')
    for k in keys:
        blobs.append(allv[k].astype(np.int8).tobytes())
    raw = b''.join(blobs)
    meta = {'smin': smin, 'smax': smax, 'keys': keys, 'spots': list(KEYS),
            'scale': 2.8, 'b64': base64.b64encode(raw).decode()}
    with open(args.out, 'w') as f:
        json.dump(meta, f, separators=(',', ':'))
    print(f'{len(keys)} configuraciones, {len(raw)} bytes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('bins', nargs='*')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--smin', type=int, default=2)
    ap.add_argument('--smax', type=int, default=15)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--eq-iters', type=int, default=1000)
    ap.add_argument('--merge', nargs='*')
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()
    merge(a) if a.merge else run(a)
