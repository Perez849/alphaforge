# AlphaForge

Predicción direccional día-siguiente con ML/redes neuronales, **con las defensas
anti-sobreajuste puestas por delante del resultado**. Entrada a T−30′ del cierre
americano para no regalarle el gap de apertura al mercado.

Estado: **39/39 autodiagnósticos superados**.

---

## Instalación

```bash
pip install -r requirements.txt
# opcional, activa la GRU real en lugar del respaldo MLP:
pip install torch
```

## Uso

```bash
# 0) SIEMPRE lo primero: comprobar que el sistema no está roto
python -m alphaforge selftest

# 1) prueba sin internet, sobre un mercado sintético con señal conocida
python -m alphaforge demo --signal mean_revert

# 2) entrenar y validar un valor real
python -m alphaforge run --ticker AAPL --start 2006-01-01 --trials 60 --predict-after

# 3) señal para mañana reutilizando el modelo guardado
python -m alphaforge predict --ticker AAPL --model af_runs/AAPL_xxxx_model.pkl

# 3b) a las 15:30 ET, con el precio que ves en pantalla
python -m alphaforge predict --ticker AAPL --model af_runs/AAPL_xxxx_model.pkl --price 231.40

# 3c) ensayo con el mercado cerrado: reconstruye la señal de una sesión pasada
python scripts/rehearse.py --back 1

# 4) barrer una cartera entera
python -m alphaforge scan --tickers AAPL,MSFT,NVDA,SPY,SAN.MC --trials 40
```

Cada `run` deja en `af_runs/`: informe en texto, informe HTML con gráficos,
JSON con todas las métricas, CSV de predicciones OOS y el modelo serializado.

Desde Python:

```python
from alphaforge import Config, run_experiment, predict_next

cfg = Config()
cfg.data.ticker = "AAPL"
cfg.data.start  = "2006-01-01"
cfg.model.n_trials = 60
res = run_experiment(cfg)
print(res.verdict["decision"], res.metrics["sharpe"], res.pbo.pbo)
print(predict_next(cfg, res))
```

---

## El problema del ancla de las 15:30 (tu requisito principal)

Querías decidir **antes** del cierre para que el gap de premercado juegue a tu
favor. Eso obliga a que el precio de entrada sea el de las 15:30 ET y a que el
objetivo se mida desde ahí:

```
y(t) = Close(t+1) / P_ancla(t) − 1
```

Ese retorno incluye los últimos 30′ de hoy + la sesión nocturna + todo mañana.
Es exactamente lo que se pierde si esperas al cierre para decidir.

**El obstáculo real:** yfinance solo sirve 60 días de barras de 30m y ~730 de 1h.
No existen 20 años de intradía gratis. La solución no es fingir que sí:

| Capa | Cobertura | Ancla |
|------|-----------|-------|
| A — histórico largo | todo | proxy `Close(t)` |
| B — histórico reciente | 60d (30m) / 730d (1h) | **real** |

Sobre el solape se mide el residuo `eps = Close/ancla − 1` (media, sigma,
autocorrelación, percentiles) y ese ruido **se inyecta en un Monte Carlo del
backtest**. El informe muestra `Sharpe bajo estrés del ancla (p05)`: si tu edge
no sobrevive a la desviación de ejecución que históricamente tienes, lo sabes.
En producción se usa siempre el ancla real, y con `--price` puedes meter a mano
el precio que ves en pantalla.

---

## Contrato temporal (aquí se muere el 90% de estos sistemas)

En el instante de decidir solo se conoce:

1. barras diarias **completas hasta t−1** → bloque BASE
2. `Open(t)` y el precio del ancla → bloque TODAY

`High(t)`, `Low(t)`, `Close(t)` y `Volume(t)` completos incluyen los últimos 30
minutos: usarlos es fuga. Implementación: `.shift(1)` **global** sobre BASE y
solo después se añade TODAY, construido contra referencias de t−1. No hay
tercera vía para meter una feature.

Encima hay un **centinela anti-fuga** que aborta la ejecución si alguna columna
correlaciona más de 0.35 con el retorno futuro, y un test que le inyecta una
fuga a propósito para verificar que salta y **nombra** la columna culpable.

---

## Features (~135 columnas, multi-timeframe)

- **Momentum**: ROC a 8 horizontes con z-score, retornos rezagados, racha, skew, kurtosis
- **Tendencia**: distancias a 6 SMA/EMA, pendientes, cruces 20/50 y 50/200
- **Osciladores**: RSI (4 periodos) y su derivada, MACD normalizado, estocástico, ADX/+DI/−DI, CCI, Williams %R, MFI
- **Volatilidad**: realizada (4 ventanas), downside, Parkinson, ratio corto/largo, ATR normalizado, Bollinger width y %B, **proxy de Hurst** (tendencia vs. reversión)
- **Estructura**: posición en rango (4 ventanas), drawdown desde máximos
- **Volumen**: relativo, z-score, pendiente OBV, dólar-volumen
- **Microestructura**: cuerpo de vela, mechas, gap, retorno intradía, overnight vs. intradía
- **Multi-timeframe**: bloque semanal y mensual resampleado, con `shift` para no usar la vela en curso
- **Cross-asset**: SPY, QQQ, VIX, TLT, UUP, GLD — retornos, beta y correlación rolling, fuerza relativa, término del VIX
- **Calendario**: codificación cíclica, turn-of-month, fin de trimestre

---

## Anti-sobreajuste — sin ponderaciones arbitrarias

| Defensa | Qué responde |
|---------|--------------|
| **Purged & embargoed walk-forward** | elimina el solape de etiquetas entre train y test; sin esto cualquier métrica está inflada |
| **PBO vía CSCV** | "de todas las configs que probé, ¿con qué probabilidad la mejor en muestra es mediocre fuera?" >0.35 → NO-GO |
| **Deflated Sharpe** | qué Sharpe exige el **puro azar** dado tu número de pruebas. Un 1.5 tras 500 intentos vale menos que un 0.8 al primero |
| **Permutación por bloques** | p-valor empírico contra un nulo que **conserva la autocorrelación** (un shuffle simple da un nulo tramposamente fácil) |
| **Calibración isotónica** | Brier skill, log-loss, ECE y curva de fiabilidad. Sin esto la "probabilidad" es un número decorativo |
| **Pesos por stacking** | logística no-negativa sobre datos out-of-sample. Cero pesos a dedo |
| **Selección de features** | poda de correlación + importancia por permutación, ajustadas **solo en train** |
| **Sample weights** | unicidad temporal (1/h), decaimiento exponencial, magnitud |

**Invariante central:** en el fold k, ningún objeto ajustado —imputador,
escalador, selector, calibrador, pesos— ha visto una fila del test de ese fold.
Se comprueba con `assert`, no con buena fe.

Los hiperparámetros se buscan **solo sobre el bloque inicial**, nunca sobre datos
OOS. Random search por familia; la combinación gana por rendimiento en CV
purgado.

---

## Modelos

Elastic-net logístico · HistGradientBoosting · ExtraTrees · MLP · **GRU en
PyTorch** sobre ventanas deslizantes (captura la *trayectoria* de los
indicadores, no solo su nivel; con early stopping y clipping de gradiente).
Si torch no está, cae a un MLP sobre la ventana aplanada — el pipeline nunca se
rompe por una dependencia ausente.

Magnitud por **regresión cuantílica** P10/P50/P90 con monotonía forzada, así la
salida es un intervalo y no un número solo.

---

## Salida

```
  P(sube)      62.5%   ALCISTA
            [########################                ]
  Retorno esperado (mediana)    0.72%
  Intervalo P10 - P90           -1.47% ... 1.68%
  Posición sugerida             +0.17x
  Fiabilidad: Sharpe OOS 2.25 | AUC 0.645 | PBO 0.103
  Veredicto del backtest: REVISAR
```

Veredicto **GO / REVISAR / NO-GO** a partir de 7 comprobaciones críticas.
Si sale NO-GO, la señal se marca como *informativa, no operable*.

---

## Prevención de bugs — 39 comprobaciones

```
features: el bloque BASE está desplazado un día
features: el bloque TODAY usa el ancla, no el cierre
features: ninguna columna correlaciona con el futuro
centinela: se detecta una fuga inyectada a propósito
etiquetas: y(t) = Close(t+1)/Anchor(t) − 1
ancla: se extrae la barra correcta de datos intradía
ancla: el volumen acumulado excluye los últimos 30 minutos
ancla: media sesión (cierre a las 13:00) no contamina el precio
etiquetas: los huecos de cotización se descartan
caché: rangos de fechas distintos no comparten fichero
preprocesado: los umbrales de recorte no ven el futuro
cotización: un precio que no es de hoy se rechaza
solapamiento: se mide y el Sharpe se corrige
splits: el embargo separa de verdad train y test
PRODUCCIÓN: predecir hoy no depende de datos de mañana
PnL: el cálculo ancla->ancla no depende del solapamiento
cotización: precios imposibles se rechazan
modelo de producción: se entrena con el histórico completo
snapshot: se avisa cuando la fila de hoy no es fiable
ensemble: la mezcla se hace en el mismo espacio en que se pesó
calibrador: se elige por validación, no a mano
pesos de muestra: el decaimiento es relativo a cada fold
selección de features: hay purga entre core y holdout
ancla: se reescala a la escala de la serie diaria
reloj: se detecta el ancla prematura, no solo la tardía
splits: purga y embargo se respetan
splits: configuración incoherente aborta
PBO: configuraciones aleatorias dan PBO ~ 0.5
PBO: una configuración genuinamente buena da PBO bajo
Deflated Sharpe: penaliza el número de pruebas
permutación: el p-valor de una métrica nula ronda 0.5
permutación por bloques: conserva la autocorrelación
métricas: Sharpe reproduce un valor analítico conocido
backtest: los costes reducen el retorno de forma monótona
backtest: sin posición no hay PnL ni comisiones
cuantiles: q10 <= q50 <= q90 siempre
datos: OHLCV corrupto se rechaza
SEÑAL: el sistema recupera una relación inyectada (AUC > 0.55)
RUIDO: el sistema NO encuentra alpha en un paseo aleatorio
```

Las dos últimas son las que de verdad importan. Resultados medidos:

| | Señal inyectada | Ruido puro |
|---|---|---|
| AUC OOS | 0.648 | 0.488 |
| Sharpe | 4.08 | −0.50 |
| PBO | 0.083 | 0.380 |
| Deflated Sharpe | 1.000 | 0.000 |
| p-valor | 0.032 | 0.871 |
| **Veredicto** | **GO** | **NO-GO (6/7 críticas fallidas)** |

Además: validación de OHLCV (precios ≤0, High<Low, saltos de split, huecos de
calendario, duplicados), guardas de división y valores no finitos, winsorización,
descarga con reintentos y caché, semillas fijadas, fingerprint SHA de la
configuración en cada artefacto, degradación elegante cuando una familia falla
(con aviso explícito, no en silencio) y `Config.validate()` que aborta ante
purga insuficiente o parámetros incoherentes.

---

## Auditoría de alineación de precios

Segunda pasada, buscando específicamente errores en el manejo de precios. Ocho
hallazgos, algunos del tipo que no se detecta nunca porque no rompe nada:

1. **Media sesión.** En Black Friday y Nochebuena el mercado cierra a las 13:00,
   no a las 16:00. El código buscaba la barra anterior a "16:00 − offset" y, al
   no existir, cogía la última disponible: **el cierre exacto**. Unas 6 veces al
   año el ancla contenía 30-60 minutos de futuro, y el volumen incluía la sesión
   entera. Ahora el margen se mide contra el cierre real de cada día.
2. **Huecos de cotización.** `y(t)` se mide contra la siguiente fila del índice.
   Si el valor estuvo suspendido dos semanas, esa etiqueta abarcaba 15 días en
   lugar de 1, mezclada con las demás como si fuera lo mismo. Ahora se descartan.
3. **Resolución con la sesión abierta.** El script que mide el acierto en vivo
   corre antes que el de señales, o sea con el mercado abierto: la última fila
   de yfinance es parcial y su "Close" es el precio del momento. Resolvía la
   predicción de ayer con un precio a media sesión. Ahora exige sesión cerrada.
4. **Cotización rancia.** Si el proveedor fallaba, el respaldo devolvía la fila
   del día anterior y se publicaba **la predicción de ayer con la fecha de hoy**.
   El error más caro posible, porque no se nota. Ahora se verifica la fecha y la
   antigüedad, y se rechaza la señal antes que publicar una falsa.
5. **Caché ciega al rango.** La clave era `TICKER__1d`: pedir desde 2006 devolvía
   la caché guardada de 2020, con años de menos y sin avisar.
6. **Look-ahead en la limpieza.** El recorte de colas usaba mediana y desviación
   de la serie completa — datos futuros respecto a cualquier fold. Poca magnitud,
   pero fuga técnica. Ahora los umbrales se ajustan solo con datos de train.
7. **Retornos solapados.** `y(t)` termina al cierre de t+1 y `y(t+1)` empieza en
   el ancla de t+1: se pisan durante la última hora. Encadenar posiciones implica
   exposición doble en esa franja y deja los retornos autocorrelacionados, lo que
   **infla el Sharpe**. Ahora se mide y se corrige con Newey-West; el veredicto
   usa el Sharpe corregido, no el bruto (en pruebas: 3.96 → 3.53).
8. **Embargo inerte.** En walk-forward anclado el train siempre precede al test,
   así que el embargo "posterior" no eliminaba ni una fila: daba una falsa
   sensación de protección. Ahora amplía la separación hacia atrás, que es donde
   sí importa con features de ventana larga.

## Tercera pasada: consistencia y contabilidad

9. **El PnL del backtest no era el ejecutable.** `y(t)` acaba al cierre de t+1 y
   `y(t+1)` empieza en el ancla de t+1: al encadenar días, la suma cuenta dos
   veces esa hora. Medido sobre una posición larga fija: **+6% de más en 3
   años**, y hasta un 30% en el acumulado de una estrategia activa. Ahora existe
   `continuous_pnl`, que mide **ancla → ancla** —sin solapamiento, lo que de
   verdad ejecutas— y se publica junto al otro. Si divergen mucho, la estrategia
   vive de un solapamiento que no puedes capturar.
10. **El modelo de producción ignoraba los meses recientes.** Se reservaba el
    18% final para calibrar y nunca se reajustaba: el modelo que opera no había
    visto los datos que mejor describen el régimen actual. Ahora se reajusta con
    todo el histórico conservando calibradores y pesos.
11. **Señales internamente contradictorias.** El clasificador decía "sube" y el
    regresor de magnitud estimaba retorno negativo, **el 12% de los días**, y se
    operaba igualmente. Ahora una puerta de coherencia deja la posición a cero
    cuando las dos cabezas discrepan.
12. **Sin control de deriva ni de calidad del snapshot.** Nada comprobaba si la
    fila de hoy venía medio vacía (el imputador la rellenaba con medianas y la
    predicción salía con la misma cara de seguridad) ni si el mercado actual se
    parecía al del entrenamiento. Ahora `snapshot_health` mide huecos, valores
    fuera del rango visto y antigüedad del modelo, y el aviso sale en el tablón.
13. **Precios imposibles aceptados.** Un cero, un split sin ajustar o un tick
    erróneo generaba un movimiento monstruoso y una señal de aspecto normal.
    Ahora se rechaza cualquier salto superior al 35% frente al cierre anterior.
14. **Apertura inventada.** Si no se conocía el Open de la sesión, se usaba el
    cierre anterior, fabricando un gap de exactamente 0% que el modelo se creía.
    Ahora queda NaN, que el imputador gestiona y, sobre todo, se ve.
15. **`git push` sin rebase.** Si el remoto había avanzado, el push fallaba y el
    día se perdía en silencio. Ahora hay `pull --rebase` con tres reintentos y
    error explícito.
16. **Modelos sin versión.** Un `.pkl` antiguo no falla al abrirse: falla más
    tarde, de formas raras, en mitad de una predicción. Ahora llevan número de
    formato y se rechazan los incompatibles.
17. **Costes inconsistentes.** El marcador en vivo usaba 5 bps fijos y el
    backtest los de la configuración: dos números no comparables. Unificados.
18. **Poda de features con datos futuros.** La cobertura mínima se evaluaba
    sobre la serie completa. Ahora sobre el primer 60%.

## Cuarta pasada: la maquinaria estadística

19. **Los pesos del ensemble se aprendían en un espacio y se aplicaban en otro.**
    `learn_ensemble_weights` ajusta una logística sobre *logits*, pero la
    predicción hacía `P @ w`, una media aritmética de *probabilidades*. Medido
    con tres modelos de calidad similar: log-loss 0.335 frente a 0.313. Peor por
    usar los pesos para una operación distinta de aquella para la que se
    optimizaron.
20. **Promediar modelos calibrados descalibra.** La media comprime la salida
    hacia 0.5 —un 8% menos de dispersión— así que la probabilidad del tablón
    salía más tibia de lo que el modelo creía, y nadie la recalibraba. Ahora la
    mezcla se recalibra sobre una mitad del bloque reservado, disjunta de la
    que fija los pesos.
21. **El calibrador estaba elegido a mano.** La isotónica necesita muchos datos;
    con los pocos cientos que quedan tras partir el bloque, sobreajusta y
    **empeora** la probabilidad (ECE de 0.017 a 0.079). Ahora se prueban Platt,
    isotónica y no calibrar, y gana el que mejore un log-loss validado.
    Resultado neto: AUC 0.621 → 0.631, ECE 0.080 → 0.050, Brier skill
    0.005 → 0.035.
22. **El decaimiento temporal miraba al final de todo el histórico.** En un fold
    que entrena hasta 2010, "reciente" debe significar reciente en 2010, no en
    2026. Con el cálculo global los folds antiguos entrenaban con pesos ~1000x
    menores. Ahora se recalculan por fold.
23. **La selección de features no purgaba.** El bloque de importancia por
    permutación pegaba con el de entrenamiento, así que la etiqueta del último
    día usaba el precio del primero del holdout. Fuga pequeña, pero situada
    justo en la elección de las features del modelo.

## Quinta pasada: hallazgos con datos reales

24. **El ancla intradía y la serie diaria estaban en escalas distintas.** Este lo
    encontró la guarda anti-fuga en la primera ejecución real, y es el mejor
    ejemplo de por qué existe. yfinance sirve el histórico diario ajustado por
    dividendos y las barras intradía con otro criterio. Como
    `y(t) = Close(t+1)/Ancla(t) − 1` mezcla ambas, aparecía un sesgo sistemático
    que ordenaba exactamente igual que el dividendo de cada valor:

    | valor | sesgo del ancla | dividendo aprox. |
    |-------|----------------|------------------|
    | SPY   | −161 bps       | 1,5 %            |
    | MSFT  | −121 bps       | 0,8 %            |
    | QQQ   | −73 bps        | 0,6 %            |
    | AAPL  | −62 bps        | 0,5 %            |
    | MU    | −36 bps        | 0,4 %            |

    Ese sesgo fabricaba una correlación de −0,35 con el retorno futuro y la
    guarda abortó el entrenamiento de SPY. No era fuga temporal: era un
    desajuste de datos que habría inflado el backtest en silencio. Ahora el
    ancla se lleva a la escala diaria usando el cierre de sesión como puente,
    lo que además anula splits y cualquier otra discrepancia de ajuste.

### El test que más importa

De todos, este es el que decide si lo demás sirve: se predice un día concreto
dos veces, con el histórico completo (que contiene su futuro) y con el histórico
cortado en ese mismo día. Las dos probabilidades salen **idénticas hasta el
último decimal**. El camino que usa el robot en vivo no mira el futuro.

Los dieciocho hallazgos tienen ya su comprobación de regresión permanente.

## Bugs reales que cazaron los tests durante el desarrollo

1. `fill_diagonal` sobre array de solo lectura (pandas 3.x)
2. `sample_weight` mal pasado a los `Pipeline` de sklearn — **tumbaba logit y MLP en silencio** dentro de un `except`
3. `rng.choice` con tuplas heterogéneas en el espacio de hiperparámetros del MLP
4. Test del ancla mal construido: barras en UTC en lugar de ET
5. Búsqueda ~20× más lenta de lo necesario: recalculaba la selección de features por trial en vez de por fold
6. `penalty='elasticnet'` y `n_jobs` deprecados en sklearn 1.8

Los cuatro primeros habrían pasado desapercibidos en producción.

---

## Parámetros que conviene tocar

| Flag | Para qué |
|------|----------|
| `--horizon open_next` | capturar **solo** el gap y cerrar en la apertura |
| `--offset 60` | decidir una hora antes del cierre |
| `--sizing kelly` | Kelly fraccional con la magnitud esperada |
| `--no-short` | solo largos (cuentas sin margen) |
| `--threshold 0.60` | operar menos y con más convicción |
| `--families hgb,gru` | limitar el zoo para ir más rápido |
| `--trials 120` | búsqueda más amplia (ojo: el DSR te penaliza por ello, y eso es lo correcto) |

---

## Aviso

Esto estima probabilidades, no certezas. Un veredicto GO significa que el edge
sobrevivió a purga, deflación, permutación y estrés de ejecución — no que vaya a
funcionar mañana. Los regímenes cambian: reentrena cada trimestre, vigila que el
Sharpe en vivo no se despegue del OOS, y dimensiona como lo que es, una apuesta
estadística con varianza.

Si sale NO-GO, lo honesto es no operarlo. Que el sistema sepa decir "aquí no hay
nada" es su función más valiosa.
