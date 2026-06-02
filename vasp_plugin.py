# vasp_plugin.py — DFT-D3/D4 dispersion via PLUGINS/FORCE_AND_STRESS
# See README.md for full tag reference, XC detection rules, and examples.

import re
import math
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

try:
    import dftd3.interface as d3
except ModuleNotFoundError:
    d3 = None

try:
    import dftd4.interface as d4
except ModuleNotFoundError:
    d4 = None

# Unit conversion (CODATA 2022)
ANGSTROM_TO_BOHR = 1e-10 / physical_constants["Bohr radius"][0]
HARTREE_TO_EV    = physical_constants["Hartree energy in eV"][0]

# IVDW → (library, fixed_damping | None)
IVDW_VERSION = {
    11: ("d3", "zero"),
    12: ("d3", "rational"),
    13: ("d4", None),
    15: ("d3", None),
}

DAMPING_MAP = {
    "zero":           "d3zero",
    "rational":       "d3bj",
    "mzero":          "d3mzero",
    "mrational":      "d3mbj",
    "optimizedpower": "d3op",
}

D3_PARAM_CLASS = {
    "d3zero":  lambda: d3.ZeroDampingParam,
    "d3bj":    lambda: d3.RationalDampingParam,
    "d3mbj":   lambda: d3.ModifiedRationalDampingParam,
    "d3mzero": lambda: d3.ModifiedZeroDampingParam,
    "d3op":    lambda: d3.OptimizedPowerDampingParam,
}

GGA_TO_METHOD = {
    "PE": "pbe", "PS": "pbesol", "RP": "rpbe", "91": "pw91",
    "MK": "revpbe", "BO": "revpbe", "OR": "revpbe",
    "RE": "revpbe", "ML": "revpbe", "CX": "revpbe",
    "AM": "am05", "B3": "b3lyp",
    "CA": "pbe", "HL": "pbe", "WI": "pbe",
}

METAGGA_TO_METHOD = {
    "R2SCAN": "r2scan", "SCAN": "scan", "RSCAN": "rscan",
    "RTPSS": "revtpss", "RTPSS0": "revtpss0",
    "TPSS": "tpss", "TPSSH": "tpssh",
    "MS2": "ms2", "MS2H": "ms2h", "PKZB": "pkzb",
    "M06L": "m06l", "MN12L": "mn12l", "MN15L": "mn15l",
}

_HYBRID_MAP: dict[tuple[str, float], str] = {
    ("PE", 0.0): "pbe0",
    ("PE", 0.2): "hse06",
    ("PE", 0.3): "hse03",
    ("PS", 0.2): "hsesol",
}
_AEXX_STANDARD = 0.25

_S9_BEHAVIOUR = {
    11: "fixed_zero",
    12: "fixed_zero",
    13: "default_one",
    15: "default_zero",
}

_D3_REQUIRED: dict[str, list[str]] = {
    "d3zero":  ["s8", "rs6"],
    "d3bj":    ["s8", "a1", "a2"],
    "d3mbj":   ["s8", "a1", "a2"],
    "d3mzero": ["s8", "rs6", "bet"],
    "d3op":    ["s8", "a1", "a2", "bet"],
}
_D4_REQUIRED: list[str] = ["s8", "a1", "a2"]

_VDW_TO_D3: dict[str, str] = {
    "VDW_S6": "s6", "VDW_S8": "s8", "VDW_S9": "s9",
    "VDW_A1": "a1", "VDW_A2": "a2",
    "VDW_SR": "rs6", "VDW_SR8": "rs8", "VDW_BETA": "bet",
}
_VDW_TO_D4: dict[str, str] = {
    "VDW_S6": "s6", "VDW_S8": "s8", "VDW_S9": "s9",
    "VDW_A1": "a1", "VDW_A2": "a2",
}
_D4_NEW_PARAM_KEYS: frozenset[str] = frozenset({"s6", "s8", "s9", "a1", "a2", "alp"})

_D3_TOML_KEY = {
    "d3bj": "bj", "d3zero": "zero",
    "d3mbj": "bjm", "d3mzero": "zerom", "d3op": "op",
}
_D3_TOML_RENAME = {
    "rs6": "rs6", "rs8": "rs8", "alp": "alp", "bet": "bet",
    "s6": "s6", "s8": "s8", "s9": "s9", "a1": "a1", "a2": "a2",
}
_TOML_META_KEYS = {"doi", "damping", "mbd"}


# ── INCAR parser ──────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"^\s*([A-Z0-9_/]+)\s*=\s*(.+)", re.IGNORECASE)


def _parse_segment(segment: str) -> tuple[str, str] | None:
    segment = segment.strip()
    if not segment or "=" not in segment:
        return None
    m = _TAG_RE.match(segment)
    return (m.group(1).upper(), m.group(2).strip()) if m else None


def parse_incar(path: str = "INCAR") -> tuple[dict, dict]:
    """Return (normal_tags, plugin_tags) from a single INCAR read."""
    normal_tags: dict[str, str] = {}
    plugin_tags: dict[str, str] = {}
    incar = Path(path)
    if not incar.exists():
        raise FileNotFoundError(f"INCAR not found: {path}")
    for raw_line in incar.read_text().splitlines():
        stripped = raw_line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            result = _parse_segment(stripped[1:].split("!")[0].strip())
            if result:
                plugin_tags[result[0]] = result[1]
        else:
            for seg in raw_line.split("!")[0].split(";"):
                result = _parse_segment(seg)
                if result:
                    normal_tags[result[0]] = result[1]
    return normal_tags, plugin_tags


# ── POTCAR parser ─────────────────────────────────────────────────────────────

_LEXCH_RE = re.compile(r"^\s*LEXCH\s*=\s*(\S+)", re.IGNORECASE)


def read_lexch_from_potcar(path: str = "POTCAR") -> str | None:
    """Return the first LEXCH value from POTCAR header, or None."""
    potcar = Path(path)
    if not potcar.exists():
        return None
    with potcar.open() as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            m = _LEXCH_RE.match(line)
            if m:
                return m.group(1).upper().strip()
    return None


# ── XC method detection ───────────────────────────────────────────────────────

def _bool_tag(value: str) -> bool:
    return value.upper().strip(".") in ("TRUE", "T")


def _float_tag(value: str) -> float:
    return float(value)


def _warn_aexx(normal_tags: dict, label: str) -> None:
    aexx = _float_tag(normal_tags.get("AEXX", "0.25"))
    if abs(aexx - _AEXX_STANDARD) > 1e-6:
        print(f"[vasp_plugin] WARNING: AEXX={aexx} for {label}. "
              f"D3/D4 params are fitted for AEXX={_AEXX_STANDARD}; "
              f"consider explicit ! VDW_* overrides.")


def detect_method(normal_tags: dict) -> str:
    """Resolve the dftd3/dftd4 method name from INCAR normal tags + POTCAR LEXCH."""
    for tag, name in (
        ("LMODELHF", "DDH"), ("LRHFCALC", "RSHXLDA/RSHXPBE"), ("LTHOMAS", "TF-screened hybrid")
    ):
        if _bool_tag(normal_tags.get(tag, ".FALSE.")):
            raise ValueError(
                f"[vasp_plugin] {tag}=.TRUE. ({name}) has no D3/D4 parameters. "
                f"Set ! VDW_S8 / ! VDW_A1 / ! VDW_A2 explicitly."
            )

    metagga = normal_tags.get("METAGGA", "").upper().strip()
    if metagga and metagga not in ("", "NONE", "FALSE", ".FALSE."):
        method = METAGGA_TO_METHOD.get(metagga)
        if method:
            lhfcalc = _bool_tag(normal_tags.get("LHFCALC", ".FALSE."))
            label   = f"METAGGA={metagga}" + (" + LHFCALC" if lhfcalc else "")
            if lhfcalc:
                _warn_aexx(normal_tags, label)
            print(f"[vasp_plugin] XC: {label} -> '{method}'")
            return method
        print(f"[vasp_plugin] WARNING: METAGGA={metagga} not in D3/D4 database; "
              f"falling back to GGA/LEXCH.")

    gga = normal_tags.get("GGA", "").upper().strip()
    if "GGA" in normal_tags and gga:
        if _bool_tag(normal_tags.get("LHFCALC", ".FALSE.")):
            hfscreen = _float_tag(normal_tags.get("HFSCREEN", "0.0"))
            method   = _HYBRID_MAP.get((gga, round(hfscreen, 1)))
            if method:
                _warn_aexx(normal_tags, f"GGA={gga} HFSCREEN={hfscreen}")
                print(f"[vasp_plugin] XC: GGA={gga} + LHFCALC + HFSCREEN={hfscreen} -> '{method}'")
                return method
            raise ValueError(
                f"[vasp_plugin] Unsupported hybrid GGA={gga} HFSCREEN={hfscreen}. "
                f"Supported: PBE0 (PE,0), HSE06 (PE,0.2), HSE03 (PE,0.3), HSEsol (PS,0.2)."
            )
        method = GGA_TO_METHOD.get(gga)
        if method:
            print(f"[vasp_plugin] XC: GGA={gga} -> '{method}'")
            return method
        print(f"[vasp_plugin] WARNING: GGA={gga} not mapped; falling back to POTCAR LEXCH.")

    lexch = read_lexch_from_potcar("POTCAR")
    if lexch:
        method = GGA_TO_METHOD.get(lexch)
        if method:
            print(f"[vasp_plugin] XC: POTCAR LEXCH={lexch} -> '{method}'")
            return method
        print(f"[vasp_plugin] WARNING: POTCAR LEXCH={lexch} not mapped.")

    print("[vasp_plugin] WARNING: XC unknown; defaulting to 'pbe'.")
    return "pbe"


# ── Module-level toml cache ───────────────────────────────────────────────────

def _load_toml_once(package_name: str) -> dict:
    if tomllib is None:
        return {}
    try:
        mod = __import__(package_name)
        with (Path(mod.__file__).parent / "parameters.toml").open("rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


_D3_TOML_DATA: dict = _load_toml_once("dftd3")
_D4_TOML_DATA: dict = _load_toml_once("dftd4")


# ── VDW parameter loading ─────────────────────────────────────────────────────

def _toml_params(data: dict, path: list[str]) -> dict:
    """Walk nested dict by path list; return {} if any key is missing."""
    node = data
    for key in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return {k: v for k, v in node.items() if k not in _TOML_META_KEYS} if isinstance(node, dict) else {}


def _load_d3_defaults(method: str, d3_version: str, s9_value: float) -> dict:
    """Load D3 defaults from cached toml; s9 set to s9_value."""
    if not _D3_TOML_DATA:
        return {}
    key = _D3_TOML_KEY[d3_version]
    merged = {
        **_toml_params(_D3_TOML_DATA, ["default", "parameter", "d3", key]),
        **_toml_params(_D3_TOML_DATA, ["parameter", method.lower(), "d3", key]),
    }
    result = {_D3_TOML_RENAME[k]: float(v) for k, v in merged.items() if k in _D3_TOML_RENAME}
    result["s9"] = s9_value
    return result


def _load_d4_defaults(method: str, s9_value: float) -> dict:
    """Load D4 defaults from cached toml; s9 set to s9_value."""
    if not _D4_TOML_DATA:
        return {}
    variant = "bj-eeq-atm" if s9_value != 0.0 else "bj-eeq-two"
    merged = {
        **_toml_params(_D4_TOML_DATA, ["default", "parameter", "d4", variant]),
        **_toml_params(_D4_TOML_DATA, ["parameter", method.lower(), "d4", variant]),
    }
    result = {k: float(v) for k, v in merged.items() if k in _D4_NEW_PARAM_KEYS}
    result["s9"] = s9_value
    return result


# Default cutoffs in Bohr — from VASP source (vdw_sd3_d4.F):
#   D3: disp2 = sqrt(9000) Bohr (~50.20 Å),  cn = 40 Bohr (~21.17 Å)
#   D4: disp2 = 60 Bohr    (~31.75 Å),        cn = 30 Bohr (~15.88 Å)
_CUTOFF_DEFAULTS_BOHR: dict[str, tuple[float, float]] = {
    "d3": (math.sqrt(9000.0), 40.0),
    "d4": (60.0,              30.0),
}


def _parse_cutoff_params(plugin_tags: dict, method_key: str) -> tuple[float, float]:
    """Return (disp2_bohr, cn_bohr). User overrides in Å converted; defaults already Bohr."""
    r_def, cn_def = _CUTOFF_DEFAULTS_BOHR[method_key]
    r  = _float_tag(plugin_tags["VDW_RADIUS"])   * ANGSTROM_TO_BOHR if "VDW_RADIUS"   in plugin_tags else r_def
    cn = _float_tag(plugin_tags["VDW_CNRADIUS"]) * ANGSTROM_TO_BOHR if "VDW_CNRADIUS" in plugin_tags else cn_def
    return r, cn


def _parse_vdw_params(
    plugin_tags: dict,
    ivdw:        int,
    d3_version:  str | None,
    method:      str = "",
) -> dict | None:
    """Return merged damping params dict for new_param(), or None if no overrides."""
    s9_beh   = _S9_BEHAVIOUR[ivdw]
    vasp_map = _VDW_TO_D3 if d3_version is not None else _VDW_TO_D4

    if s9_beh == "fixed_zero" and "VDW_S9" in plugin_tags:
        print(f"[vasp_plugin] WARNING: ! VDW_S9 ignored for ! IVDW={ivdw} (s9=0 fixed).")

    user: dict[str, float] = {}
    for vtag, kwarg in vasp_map.items():
        if vtag == "VDW_S9" and s9_beh == "fixed_zero":
            continue
        if vtag in plugin_tags:
            user[kwarg] = _float_tag(plugin_tags[vtag])

    if s9_beh == "fixed_zero":
        s9 = 0.0
    elif s9_beh == "default_one":
        s9 = user.pop("s9", 1.0)
    else:
        s9 = user.pop("s9", 0.0)

    if not user and s9 == (1.0 if s9_beh == "default_one" else 0.0):
        return None

    defaults = _load_d3_defaults(method, d3_version, s9) if d3_version else _load_d4_defaults(method, s9)
    merged   = {**defaults, **user}
    merged["s9"] = s9

    required = _D3_REQUIRED[d3_version] if d3_version else _D4_REQUIRED
    missing  = [r for r in required if r not in merged]
    if missing:
        vasp_missing = [k for k, v in vasp_map.items() if v in missing and k != "VDW_S9"]
        raise ValueError(
            f"[vasp_plugin] Required params missing after merging defaults for '{method}'. "
            f"Functional may not be in the D3/D4 database. "
            f"Set explicitly: {vasp_missing}"
        )
    return merged


# ── Dispersion runners ────────────────────────────────────────────────────────

def _run_d3(
    version:         str,
    method:          str,
    numbers,
    positions,
    lattice_vectors,
    custom_params:   dict | None         = None,
    cutoffs:         tuple[float, float] = (math.sqrt(9000.0), 40.0),
) -> dict:
    if d3 is None:
        raise ModuleNotFoundError("dftd3 not installed: conda install -c conda-forge dftd3-python")
    cls   = D3_PARAM_CLASS[version]()
    param = cls(**custom_params) if custom_params else cls(method=method, atm=False)
    disp  = d3.DispersionModel(numbers=numbers, positions=positions, lattice=lattice_vectors)
    disp.set_realspace_cutoff(cutoffs[0], cutoffs[0], cutoffs[1])
    return disp.get_dispersion(param=param, grad=True)


def _run_d4(
    method:          str,
    numbers,
    positions,
    lattice_vectors,
    d4_model:        str                 = "d4",
    custom_params:   dict | None         = None,
    cutoffs:         tuple[float, float] = (60.0, 30.0),
) -> dict:
    if d4 is None:
        raise ModuleNotFoundError("dftd4 not installed: conda install -c conda-forge dftd4-python")
    param = d4.DampingParam(**custom_params) if custom_params else d4.DampingParam(method=method, atm=True)
    disp  = d4.DispersionModel(numbers=numbers, positions=positions, lattice=lattice_vectors, model=d4_model)
    disp.set_realspace_cutoff(cutoffs[0], cutoffs[0], cutoffs[1])
    return disp.get_dispersion(param=param, grad=True)


# ── Plugin entry point ────────────────────────────────────────────────────────

def force_and_stress(constants, additions) -> None:
    """PLUGINS/FORCE_AND_STRESS entry point — called by VASP at each ionic step."""
    normal_tags, plugin_tags = parse_incar("INCAR")

    ivdw = int(plugin_tags.get("IVDW", "0"))
    if ivdw == 0:
        print("[vasp_plugin] ! IVDW not found; no dispersion applied.")
        return
    if ivdw not in IVDW_VERSION:
        raise ValueError(f"[vasp_plugin] ! IVDW={ivdw} unsupported. Valid: {sorted(IVDW_VERSION)}")

    version_key, fixed_damping = IVDW_VERSION[ivdw]
    method = detect_method(normal_tags)

    lattice_vectors = np.asarray(constants.lattice_vectors) * ANGSTROM_TO_BOHR
    positions       = np.asarray(constants.positions) @ lattice_vectors
    numbers         = np.asarray(constants.atomic_numbers)[np.asarray(constants.ion_types)]

    if version_key == "d3":
        if fixed_damping is not None:
            damping_str = fixed_damping
        else:
            if "SDFTD3_DAMPING" not in plugin_tags:
                raise ValueError(
                    f"[vasp_plugin] ! IVDW=15 requires ! SDFTD3_DAMPING. "
                    f"Valid: {list(DAMPING_MAP)}"
                )
            damping_str = plugin_tags["SDFTD3_DAMPING"].lower().strip()
        if damping_str not in DAMPING_MAP:
            raise ValueError(f"[vasp_plugin] Unknown ! SDFTD3_DAMPING='{damping_str}'. Valid: {list(DAMPING_MAP)}")
        d3_version    = DAMPING_MAP[damping_str]
        custom_params = _parse_vdw_params(plugin_tags, ivdw=ivdw, d3_version=d3_version, method=method)
        cutoffs       = _parse_cutoff_params(plugin_tags, "d3")
        print(f"[vasp_plugin] D3 {d3_version}  method={method}"
              + (f"  overrides={custom_params}" if custom_params else ""))
        res = _run_d3(d3_version, method, numbers, positions, lattice_vectors,
                      custom_params=custom_params, cutoffs=cutoffs)

    else:
        raw_model = plugin_tags.get("DFTD4_MODEL", "D4").upper().strip()
        if raw_model not in ("D4", "D4S"):
            raise ValueError(f"[vasp_plugin] Unknown ! DFTD4_MODEL='{raw_model}'. Valid: D4, D4S")
        d4_model      = raw_model.lower()
        custom_params = _parse_vdw_params(plugin_tags, ivdw=ivdw, d3_version=None, method=method)
        cutoffs       = _parse_cutoff_params(plugin_tags, "d4")
        print(f"[vasp_plugin] D4 model={d4_model}  method={method}"
              + (f"  overrides={custom_params}" if custom_params else ""))
        res = _run_d4(method, numbers, positions, lattice_vectors,
                      d4_model=d4_model, custom_params=custom_params, cutoffs=cutoffs)

    additions.total_energy += res["energy"]   * HARTREE_TO_EV
    additions.forces       -= res["gradient"] * (HARTREE_TO_EV * ANGSTROM_TO_BOHR)
    additions.stress       += -res["virial"]  * HARTREE_TO_EV
