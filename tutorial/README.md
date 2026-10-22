# salvador tutorial

Abundances of the red giant 1500138 of the globular cluster M54 from its GIRAFFE HR18 spectrum,
taken from the ESO archive.

| File | Content |
| --- | --- |
| `input.salvador` | settings: NORM = 2 (continuum windows), automatic mode, NPROC = 7 |
| `lista.stars` | the spectrum and its parameters (T<sub>eff</sub> = 4248 K, log g = 0.77, [M/H] = −1.54, alpha code 1, v<sub>t</sub> = 1.95 km/s, RV = 133.79 km/s) |
| `M54_HR18_1500138.fits` | the spectrum |
| `linelist.txt` | 27 lines (the first two are outside the spectrum and are skipped) |
| `abu_var.txt` | abundance variations of the model (empty) |
| `wind_continuum_hr18.txt` | continuum windows |
| `linelist.xkur` | the Kurucz linelists for K2: **edit the paths for your computer** |
| `expected/` | the results of a reference run |

Run:

```bash
salvador
```

Then compare with the reference run:

```bash
mkdir -p regression_ref && cp expected/M54_HR18_1500138.fit expected/abundance_final.txt regression_ref/
salvador-regression
```

`lista.stars` has 12 columns: the errors due to the atmospheric parameters need three values
(ΔT<sub>eff</sub>, Δlog g, Δv<sub>t</sub>), so salvador prints a warning and does not compute them.
Add a third value (13 columns) to compute them.
