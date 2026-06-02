# Mimic of `subdft_sd3_d4.F`

DFT-D3 / DFT-D4 dispersion correction via `PLUGINS/FORCE_AND_STRESS`.  
GPU-compatible: runs on CPU, independent of the VASP binary compiler.

## Dependencies

```bash
conda install -c conda-forge dftd3-python dftd4-python scipy
```

`tomllib` (Python ≥ 3.11) or `tomli` (backport) is used for partial VDW
parameter overrides. Without it, overrides still work but all required params
must be set explicitly.

```bash
conda install -c conda-forge tomli   # only needed for Python < 3.11
```

## INCAR setup

Two classes of lines:

| Line type | Who reads it | Tags |
|-----------|-------------|------|
| Normal (no `!`) | VASP **and** plugin | `GGA`, `METAGGA`, `LHFCALC`, `HFSCREEN`, `AEXX` |
| Plugin (`!`-prefixed) | Plugin **only** | All tags listed below |

Each plugin tag must be on its **own** `!`-prefixed line. Do not combine two
plugin tags on one line — the second `!` is treated as an inline comment and
the second tag is silently dropped.

Required in INCAR:

```fortran
PLUGINS/FORCE_AND_STRESS = T
```

## Plugin-only tags

### Method control

```fortran
! IVDW = 11              ! D3 zero-damping (Grimme 2010)
! IVDW = 12              ! D3 Becke-Johnson damping (Grimme 2011)
! IVDW = 13              ! D4 (DFTD4_MODEL defaults to D4)
! DFTD4_MODEL = D4S      ! D4S smoothed model — only with ! IVDW = 13
! IVDW = 15              ! simple-DFT-D3 — ! SDFTD3_DAMPING required
! SDFTD3_DAMPING = zero | rational | mzero | mrational | optimizedpower
```

### Damping parameter overrides (all optional)

Unset parameters use library defaults for the detected XC functional.

| Tag | dftd3 kwarg | dftd4 kwarg | Applies to |
|-----|------------|------------|------------|
| `! VDW_S6` | `s6` | `s6` | all |
| `! VDW_S8` | `s8` | `s8` | all |
| `! VDW_S9` | `s9` | `s9` | all — see s9 defaults below |
| `! VDW_A1` | `a1` | `a1` | rational, mrational, optimizedpower |
| `! VDW_A2` | `a2` | `a2` | rational, mrational, optimizedpower |
| `! VDW_SR` | `rs6` | — | zero, mzero |
| `! VDW_SR8` | `rs8` | — | zero, mzero |
| `! VDW_BETA` | `bet` | — | mzero, optimizedpower |

**s9 (ATM three-body term) defaults:**

| `! IVDW` | s9 behaviour |
|----------|-------------|
| 11, 12 | fixed `0.0` — `! VDW_S9` ignored with warning |
| 13 | default `1.0` — `! VDW_S9` overrides |
| 15 | default `0.0` — `! VDW_S9` overrides |

### Cutoff overrides (optional, in Å)

| Tag | Default D3 | Default D4 |
|-----|-----------|-----------|
| `! VDW_RADIUS` | √9000 Bohr (≈ 50.20 Å) | 60 Bohr (≈ 31.75 Å) |
| `! VDW_CNRADIUS` | 40 Bohr (≈ 21.17 Å) | 30 Bohr (≈ 15.88 Å) |

## XC detection priority

Reads from normal INCAR lines + POTCAR `LEXCH`:

1. `LMODELHF` / `LRHFCALC` / `LTHOMAS` = `.TRUE.` → **ValueError** (no D3/D4 params)
2. `METAGGA` (overrides GGA)
3. `GGA` + `LHFCALC` → hybrid lookup by `(GGA, HFSCREEN)`:

   | GGA | HFSCREEN | Functional |
   |-----|---------|------------|
   | `PE` | `0.0` (absent) | PBE0 |
   | `PE` | `0.2` | HSE06 |
   | `PE` | `0.3` | HSE03 |
   | `PS` | `0.2` | HSEsol |

   Any other combination raises **ValueError**. PBEsol0 (`PS`, `0.0`) is
   excluded — no dedicated D3/D4 parameter set exists; set `! VDW_S8` /
   `! VDW_A1` / `! VDW_A2` explicitly.

4. `GGA` alone (plain GGA)
5. `POTCAR` `LEXCH` (fallback when `GGA` absent from INCAR)
6. `"pbe"` + warning (last resort)

**AEXX:** D3/D4 hybrid parameters are fitted for `AEXX=0.25`. A warning is
printed if `AEXX` differs, but the method name is still returned.

## Example INCAR snippets

### PBE-D3(BJ)

```fortran
GGA                      = PE
PLUGINS/FORCE_AND_STRESS = T
! IVDW = 12
```

### HSE06-D3(BJ) with explicit parameters

```fortran
GGA                      = PE
LHFCALC                  = .TRUE.
HFSCREEN                 = 0.2
PLUGINS/FORCE_AND_STRESS = T
! IVDW = 12
! VDW_S8 = 2.310
! VDW_A1 = 0.383
! VDW_A2 = 5.685
```

### r²SCAN-D4

```fortran
METAGGA                  = R2SCAN
PLUGINS/FORCE_AND_STRESS = T
! IVDW = 13
```

### simple-DFT-D3 modified BJ with ATM

```fortran
GGA                      = PE
PLUGINS/FORCE_AND_STRESS = T
! IVDW = 15
! SDFTD3_DAMPING = mrational
! VDW_S9 = 1.0
```

## Units

| Quantity | VASP / plugin receives | dftd3/dftd4 expects | Conversion |
|----------|----------------------|---------------------|------------|
| Lattice vectors | Å | Bohr | `× ANGSTROM_TO_BOHR` (CODATA 2022) |
| Positions | fractional (dimensionless) | Cartesian Bohr | `frac @ lattice_bohr` |
| Energy | eV | Hartree | `× HARTREE_TO_EV` |
| Forces | eV/Å | Ha/Bohr | `× HARTREE_TO_EV × ANGSTROM_TO_BOHR`; sign: force = −gradient |
| Stress | eV (raw virial) | Hartree | `× HARTREE_TO_EV`; sign: stress += −virial |

## VASP plugin interface layout

Confirmed from `ConstantsForceAndStress` diagnostics:

| Field | Shape | dtype | Unit / note |
|-------|-------|-------|-------------|
| `lattice_vectors` | `(3, 3)` | float64 | Å, row vectors |
| `positions` | `(N, 3)` | float64 | fractional (Direct), dimensionless |
| `atomic_numbers` | `(n_species,)` | int32 | atomic number per species |
| `ion_types` | `(N,)` | int32 | per-atom species index, 0-based |

Coordinate conversion to Cartesian Bohr (matches `vdw_sd3_d4.F`):

```python
lattice_bohr   = lattice_vectors * ANGSTROM_TO_BOHR   # (3,3) Bohr, row vectors
positions_bohr = positions @ lattice_bohr              # (N,3) Cartesian Bohr
numbers        = atomic_numbers[ion_types]             # (N,) per-atom Z
```
