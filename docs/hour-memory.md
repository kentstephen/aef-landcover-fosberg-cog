# Does the weather's memory matter

Question: should the ground stay lit after a bad hour (the afterglow, holding by land cover), or does only the
hour count? Script: `ignition/hour_memory.py`. Data: the 22,204 ignition hours of the big set against the same
points' other hours (the t_ctrl control: same pixel, another hour in the same 90-day block, more than a day
away), so the ground cancels and only the hour differs. Logistic fits, five folds held out by HRRR square,
weights per standard deviation.

| hour features | AUC | weights per sd |
|---|---|---|
| FFWI this hour | 0.656 | ffwi +0.55 |
| past day's peak FFWI | 0.582 | max24 +0.26 |
| FFWI + past day's peak | 0.655 | ffwi +0.61, max24 -0.10 |
| FFWI + rain past 72 h | 0.662 | ffwi +0.51, precip72 -0.28 |
| FFWI + peak + rain | 0.664 | ffwi +0.60, max24 -0.15, precip72 -0.30 |
| FFWI + T + RH + wind | 0.706 | ffwi -0.14, T +0.50, RH -0.42, wind +0.34 |
| the same + past day's peak | 0.711 | T +0.49, RH -0.48, wind +0.35, max24 -0.21 |
| wind alone | 0.591 | wind +0.32 |
| RH alone | 0.659 | RH -0.62 |

Reading:

- The hour is the signal. This hour's FFWI alone ranks an ignition hour against another hour at the same
  point two times in three; temperature, humidity and wind together, seven in ten.
- Memory does not add. Given this hour, the past day's peak carries a small negative weight: a point whose
  worst hour was earlier and is calmer now is slightly less likely to ignite now, not more. Rain in the past
  three days subtracts (a quarter of a standard deviation per sd of rain).
- So the map's afterglow (the ground holding a past hour's peak by land cover) has no support in the ignition
  record. It defaults to off. The fuel moisture lag argument (fine fuel follows the air within hours, timber
  over a day) may be true of the fuel and still not show in ignitions, because the hour's humidity and
  temperature already carry it.
- How much should move: at one point, FFWI swings between its hours with a standard deviation of 6.9; across
  points at the ignition hour the spread is 14.1. The hour moves the picture about half as much as place does.
  The median ignition hour has FFWI 16.5 against 10.2 for the same point's other hours. A still picture is
  wrong; a picture that changes everywhere is also wrong; the fuel pattern should stay and the lit share of it
  should breathe with the hour.

Limits: the HRRR analysis at 3 km and the top of the hour; no fuel moisture product in the fit (the past day's
peak and the rain stand in for it); the same reporting gaps as the rest of the WFIGS work.
