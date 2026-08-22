# Puesta en marcha en GitHub

Objetivo: entras a `https://TUUSUARIO.github.io/alphaforge` desde el móvil a las
21:05 y ves el tablón del día.

## Cómo encaja todo

GitHub Pages **solo sirve archivos**, no ejecuta Python. Quien ejecuta es
**GitHub Actions**. La cadena es:

```
Actions (sábado)   -> entrena los modelos      -> models/*.pkl
Actions (L-V 15:00 ET) -> lee el precio del momento -> docs/data/latest.json
Actions (antes de eso) -> resuelve el día anterior  -> docs/data/scoreboard.json
GitHub Pages       -> muestra docs/index.html leyendo esos JSON
```

Todo gratis si el repositorio es **público** (Actions ilimitado). En privado
gastarías del cupo de 2.000 min/mes: el entrenamiento semanal se lo come.

## Pasos

**1. Sube el repositorio**

```bash
git init && git add . && git commit -m "alphaforge"
git remote add origin https://github.com/TUUSUARIO/alphaforge.git
git push -u origin main
```

**2. Activa Pages**

Settings → Pages → Source: *Deploy from a branch* → rama `main`, carpeta
`/docs` → Save.

**3. Permite que el robot escriba**

Settings → Actions → General → Workflow permissions → **Read and write
permissions** → Save. Sin esto el robot calcula las señales pero no puede
publicarlas.

**4. Elige tus valores**

Edita `universe.json`. Empieza con pocos: cada uno tarda entre 3 y 8 minutos en
entrenarse.

**5. Entrena por primera vez**

Actions → *Reentrenamiento semanal* → Run workflow. Tarda entre 30 min y 1 hora
para 8 valores. Antes de entrenar ejecuta el autodiagnóstico: si algo está roto,
no entrena nada.

**6. Prueba la publicación**

Actions → *Señales diarias* → Run workflow, marcando `force` (así funciona
aunque el mercado esté cerrado). Luego abre tu página.

A partir de ahí va solo.

## La hora exacta

Quieres las 15:00 ET, una hora antes del cierre. Tres complicaciones resueltas:

- **Horario de verano.** 15:00 ET son las 19:00 UTC de marzo a noviembre y las
  20:00 el resto. Se programan los dos crons; el script mira la hora real de
  Nueva York y el que no toca sale sin hacer nada (código 78, que Actions no
  cuenta como fallo).
- **El cron de GitHub llega tarde**, entre 1 y 20 minutos, siempre. Por eso se
  lanza a menos 15 y el script **espera** hasta el instante exacto antes de leer
  el precio (`--wait`).
- **Festivos.** No hay calendario codificado a mano, que envejece mal: si el
  mercado está cerrado, el script lo detecta y no publica.

Si el precio se captura fuera de plazo, el tablón lo marca con **TOMADA TARDE**
en lugar de disimularlo.

## Probar en fin de semana o con el mercado cerrado

El robot de señales **no funciona** fuera de sesión, y es a propósito: sin
barras del día en curso el respaldo devuelve el precio de la sesión anterior, y
publicar eso sería emitir la predicción de ayer con la fecha de hoy. Se rechaza.

Tampoco se reconstruye "como si" se hubiera ejecutado a la hora correcta de un
día pasado. Técnicamente se podría —yfinance guarda 60 días de barras de 30
minutos— pero esa señal ya no es operable, y meterla en el marcador lo
convertiría en otra cosa: el marcador solo vale porque contiene señales
publicadas antes de conocerse el resultado.

Lo que sí puedes hacer con el mercado cerrado:

**Entrenar.** Funciona cualquier día y es lo que llena la pestaña *Fiabilidad*
con años de predicciones fuera de muestra. Es el grueso de lo que quieres ver.

**Ensayar el camino de producción:**

```bash
python scripts/rehearse.py --back 1      # la última sesión cerrada
python scripts/rehearse.py --back 5      # hace cinco sesiones
```

Reconstruye el ancla intradía real de esa sesión, genera la señal que se habría
publicado y, si la sesión siguiente ya cerró, dice si acertó y si el movimiento
cayó dentro del rango P10–P90. Sirve para comprobar que la maquinaria entera
funciona con datos reales sin esperar al lunes. **Solo imprime por pantalla: no
escribe nada ni toca el marcador.**

Un aviso sobre cómo leerlo: el acierto de una sola sesión no dice nada. Con
cuatro valores, acertar tres pasa por azar una de cada tres veces.

## El retardo de los datos, que sí importa

yfinance sirve precios con retardo variable. Un desfase de minutos cambia el
precio de entrada y por tanto la señal. Para mirar el tablón sirve; para operar
de verdad, coge el precio de tu bróker y pásalo a mano:

```bash
python -m alphaforge predict --ticker AAPL --model models/AAPL.pkl --price 231.40
```

## Coste de tiempo

| Tarea | Frecuencia | Duración |
|---|---|---|
| Entrenamiento | semanal | 3-8 min por valor |
| Señales | diaria | ~20 s por valor |
| Resolución | diaria | ~10 s |

Con 8 valores: unos 45 min los sábados y 4 min al día.

## Fiabilidad desde el primer día

No hay que esperar dos meses. En cuanto termina el **primer entrenamiento**, la
pestaña *Fiabilidad* ya está llena, porque el walk-forward produce miles de
predicciones fuera de muestra: cada día del histórico fue predicho por un modelo
entrenado solo con datos anteriores.

Ahí verás, por cada valor:

- **Acierto direccional** sobre años de predicciones (50% = moneda al aire)
- **Curva de resultados** frente a comprar y mantener
- **Curva de calibración** — la prueba más directa que existe: cuando dice 60%,
  ¿sube el 60% de las veces? Marca ámbar = lo prometido, barra = lo cumplido
- **Año a año** — si el resultado se concentra en años antiguos, el edge se apagó
- **PBO, Sharpe deflactado y p-valor** con su umbral al lado
- **Las 9 comprobaciones** con su detalle
- Una **cartera equiponderada** de los modelos que pasaron la validación

Eso es una medida legítima de fiabilidad, con dos salvedades que el tablón no
esconde: los hiperparámetros se eligieron una vez mirando el bloque inicial, y en
la parte antigua del histórico el precio de referencia es un proxy. Por eso el
**marcador en vivo** se mantiene aparte: mide señales publicadas antes de
conocerse el resultado, y ese sí es imposible de retocar.

## Qué mirar en el tablón

Tres bloques:

**Pestaña Hoy.** Las señales del día: ticker, probabilidad y la barra de rango.
Las filas atenuadas no superaron la validación. La barra muestra el rango
probable (P10–P90) con la estimación central marcada; verás que el rango es mucho
más ancho que la estimación, y eso es la realidad, no un defecto.

**Pestaña Fiabilidad.** Lo descrito arriba, disponible desde el primer día.

Si tras un par de meses el acierto en vivo ronda el 50%, el sistema te lo estará
diciendo claramente y lo sensato es hacerle caso.

## Ajustes útiles en `universe.json`

| Cambio | Efecto |
|---|---|
| `"anchor_offset_min": 30` | decidir media hora antes del cierre |
| `"horizon": "open_next"` | capturar solo el gap y cerrar en la apertura |
| `"prob_threshold": 0.60` | operar menos y solo con convicción alta |
| `"allow_short": false` | solo largos |
| `"n_trials": 80` | búsqueda más amplia (y listón estadístico más alto) |

## Si algo falla

| Síntoma | Causa |
|---|---|
| «sin modelo entrenado» | falta ejecutar *Reentrenamiento semanal* |
| «faltan N features» | el modelo es de una versión anterior: reentrena |
| No se publica nada | permisos de escritura sin activar (paso 3) |
| Sale código 78 | mercado cerrado: es lo correcto, no un error |
| «PRECIO NO INTRADÍA» | no se pudo leer el precio del momento y se usó el diario |
