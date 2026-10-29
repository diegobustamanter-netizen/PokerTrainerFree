# Poker Trainer Free

Entrenador gratuito de poker para Sit & Go hyper-turbo de 3 jugadores (Spin & Go, Twister, Expresso...) y heads-up. Al abrir la página eliges el modo:

- **Manos al azar:** situaciones sueltas de push/fold y postflop (flop, turn y river) con corrección inmediata y estadísticas por spot.
- **Jugar partidas:** un Sit & Go completo contra dos rivales técnicos. Con 10 bb o menos juegan push/fold Nash; con más fichas abren con min-raise y juegan postflop apostando por valor y casi sin faroles. Tus decisiones de push/fold se comparan con Nash y las postflop con tu equity contra el rango estimado del rival frente a las pot odds.

Toda la aplicación es un solo archivo, `index.html`, sin servidor ni dependencias.

## Publicarlo en GitHub Pages

1. Crea un repositorio nuevo en GitHub (por ejemplo `PokerTrainerFree`).
2. Sube los archivos de esta carpeta (como mínimo `index.html`).
3. En el repositorio ve a **Settings > Pages**, en *Source* elige **Deploy from a branch**, rama `main` y carpeta `/ (root)`.
4. En uno o dos minutos la página queda en `https://<tu-usuario>.github.io/PokerTrainerFree/`.

## Estructura

```
index.html          La página final (template + datos ya incluidos)
template.html       La página sin datos, para editar el diseño
build.py            Inserta los rangos (solver/data.json y solver/unequal.json) en template.html y genera index.html
solver/sampler.c    Simulador Monte Carlo: equity heads-up y 3-way por trío de manos
solver/solver.py    Solver Nash (fictitious play) para 3 jugadores y heads-up, stacks iguales
solver/solver_unequal.py  El mismo solver con stacks distintos (side pots incluidos)
```

## Regenerar los rangos

Requiere `gcc`, Python 3 y `numpy`. La simulación usa unos 120 MB de RAM por proceso y el solver alrededor de 1 GB.

```bash
cd solver
gcc -O3 -march=native -o sampler sampler.c
./sampler 75000000 101 s1.bin      # 75 millones de repartos (~1 min)
./sampler 75000000 202 s2.bin      # otra semilla, se suman en el solver
python3 solver.py s1.bin s2.bin -o data.json --iters 1500
# stacks distintos: 287 configuraciones, en 2 procesos (~35 min)
python3 solver_unequal.py s1.bin s2.bin --part 0 --parts 2 -o part0.npz &
python3 solver_unequal.py s1.bin s2.bin --part 1 --parts 2 -o part1.npz &
wait
python3 solver_unequal.py --merge part0.npz part1.npz -o unequal.json
cd ..
python3 build.py
```

## Supuestos del modelo

- Ciegas 0.5 / 1 bb, sin ante, stacks de 2 a 15 bb (iguales o distintos).
- Con stacks distintos, cada stack se reduce a min(propio, mayor stack rival): el resto nunca se juega en chipEV.
- Solo all-in o fold (y pagar o fold frente a un all-in).
- Se maximiza chipEV, que en torneos winner-takes-all equivale al $EV.

## Ideas para seguir

- Sit & Go de 5 jugadores con ICM cuando paga más de un puesto.
- Antes y min-raise como opción adicional.
