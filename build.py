"""Inserta los rangos dentro de template.html y genera index.html.

  solver/data.json     stacks iguales y heads-up  (solver.py)
  solver/unequal.json  stacks distintos            (solver_unequal.py --merge)

Uso: python3 build.py
"""
from pathlib import Path

root = Path(__file__).parent
html = (root / 'template.html').read_text(encoding='utf-8')
data = (root / 'solver' / 'data.json').read_text()
unequal_file = root / 'solver' / 'unequal.json'
unequal = unequal_file.read_text() if unequal_file.exists() else 'null'
for marker in ('/*__DATA__*/null', '/*__UDATA__*/null'):
    assert marker in html, marker
html = html.replace('/*__DATA__*/null', data).replace('/*__UDATA__*/null', unequal)
(root / 'index.html').write_text(html, encoding='utf-8')
print('index.html generado' + ('' if unequal != 'null' else ' (sin stacks distintos)'))
