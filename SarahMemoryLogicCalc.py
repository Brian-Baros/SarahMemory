"""
--== SarahMemory Project ==--
File: SarahMemoryLogicCalc.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-02-18
Time: 10:11:54
Author: © 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================

PURPOSE:
Rational Core Engine for SarahMemory AiOS:
- Deterministic scientific calculator + engineering solver
- Dimensional analysis + unit normalization
- Logic inference + formula selection
- Semantic Interlingua (formal meaning representation)
- Natural Language Generation (explanations)
- Translation via interlingua (“math/logic as Rosetta Stone”)

DESIGN GOALS:
- Local-first, deterministic, auditable
- Hard-coded formulas + rules, expandable without refactors
- Output is structured + human-readable
"""

from __future__ import annotations

import math
import re
import time
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

Number = Union[int, float]


# =============================================================================
# UTILITIES
# =============================================================================

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _clip(s: str, n: int = 500) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "..."

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _lower(s: str) -> str:
    return (s or "").strip().lower()


# =============================================================================
# SEMANTIC INTERLINGUA (MEANING LAYER)
# =============================================================================

@dataclass
class Term:
    """Atomic term used in propositions: variable, constant, entity, unit."""
    name: str
    kind: str = "symbol"  # symbol|number|entity|unit|concept
    value: Optional[Any] = None

@dataclass
class Proposition:
    """
    Typed proposition:
    - predicate: e.g., "equals", "causes", "has", "increases", "converts_to"
    - args: terms
    - meta: confidence, units, domain, derivation
    """
    predicate: str
    args: List[Term] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MeaningGraph:
    """Bundle of propositions representing meaning for reasoning and NLG."""
    props: List[Proposition] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, predicate: str, args: List[Term], **meta: Any) -> None:
        self.props.append(Proposition(predicate=predicate, args=args, meta=meta))

    def summarize(self) -> Dict[str, Any]:
        return {
            "props": [
                {
                    "predicate": p.predicate,
                    "args": [{"name": t.name, "kind": t.kind, "value": t.value} for t in p.args],
                    "meta": p.meta
                }
                for p in self.props
            ],
            "meta": self.meta
        }


# =============================================================================
# UNIT SYSTEM + DIMENSIONAL ANALYSIS (ENTERPRISE-GRADE SAFETY)
# =============================================================================

@dataclass(frozen=True)
class Dimension:
    """
    Simple dimensional vector:
    L (length), M (mass), T (time), I (current), Th (temperature), N (amount), J (luminous)
    """
    L: int = 0
    M: int = 0
    T: int = 0
    I: int = 0
    Th: int = 0
    N: int = 0
    J: int = 0

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            L=self.L + other.L, M=self.M + other.M, T=self.T + other.T,
            I=self.I + other.I, Th=self.Th + other.Th, N=self.N + other.N, J=self.J + other.J
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            L=self.L - other.L, M=self.M - other.M, T=self.T - other.T,
            I=self.I - other.I, Th=self.Th - other.Th, N=self.N - other.N, J=self.J - other.J
        )

    def __pow__(self, p: int) -> "Dimension":
        return Dimension(
            L=self.L * p, M=self.M * p, T=self.T * p,
            I=self.I * p, Th=self.Th * p, N=self.N * p, J=self.J * p
        )

    def as_tuple(self) -> Tuple[int, int, int, int, int, int, int]:
        return (self.L, self.M, self.T, self.I, self.Th, self.N, self.J)


@dataclass(frozen=True)
class Unit:
    name: str
    symbol: str
    factor_to_si: float  # multiply to SI base
    dim: Dimension

class UnitRegistry:
    """
    Minimal SI-focused registry. Extend aggressively over time.
    This is the control plane for dimensional correctness.
    """
    def __init__(self) -> None:
        self._by_name: Dict[str, Unit] = {}
        self._by_symbol: Dict[str, Unit] = {}

        # Base dims
        L = Dimension(L=1)
        M = Dimension(M=1)
        T = Dimension(T=1)
        I = Dimension(I=1)
        Th = Dimension(Th=1)
        N = Dimension(N=1)

        # Base units
        self.add(Unit("meter", "m", 1.0, L))
        self.add(Unit("kilometer", "km", 1000.0, L))
        self.add(Unit("centimeter", "cm", 0.01, L))
        self.add(Unit("millimeter", "mm", 0.001, L))
        self.add(Unit("inch", "in", 0.0254, L))
        self.add(Unit("foot", "ft", 0.3048, L))
        self.add(Unit("mile", "mi", 1609.344, L))

        self.add(Unit("kilogram", "kg", 1.0, M))
        self.add(Unit("gram", "g", 0.001, M))
        self.add(Unit("pound", "lb", 0.45359237, M))

        self.add(Unit("second", "s", 1.0, T))
        self.add(Unit("minute", "min", 60.0, T))
        self.add(Unit("hour", "h", 3600.0, T))

        self.add(Unit("ampere", "A", 1.0, I))

        # Derived dims
        # Newton: kg*m/s^2
        self.add(Unit("newton", "N", 1.0, M * L / (T ** 2)))
        # Joule: N*m
        self.add(Unit("joule", "J", 1.0, (M * L / (T ** 2)) * L))
        # Watt: J/s
        self.add(Unit("watt", "W", 1.0, ((M * L / (T ** 2)) * L) / T))
        # Pascal: N/m^2
        self.add(Unit("pascal", "Pa", 1.0, (M * L / (T ** 2)) / (L ** 2)))
        self.add(Unit("kilopascal", "kPa", 1000.0, (M * L / (T ** 2)) / (L ** 2)))
        self.add(Unit("bar", "bar", 100000.0, (M * L / (T ** 2)) / (L ** 2)))
        self.add(Unit("psi", "psi", 6894.757293168, (M * L / (T ** 2)) / (L ** 2)))

        self.add(Unit("volt", "V", 1.0, (M * (L ** 2)) / (T ** 3) / I))
        self.add(Unit("ohm", "Ω", 1.0, (M * (L ** 2)) / (T ** 3) / (I ** 2)))

        # Temperature handled
        # Robotics / kinematics convenience units (SI-compatible)
        self.add(Unit("radian", "rad", 1.0, Dimension()))  # dimensionless
        self.add(Unit("degree", "deg", math.pi / 180.0, Dimension()))  # dimensionless

        # Common composite kinematic units (treated as named units to keep conversion deterministic)
        self.add(Unit("meters_per_second", "m/s", 1.0, L / T))
        self.add(Unit("meters_per_second2", "m/s^2", 1.0, L / (T ** 2)))
        self.add(Unit("newton_meter", "N*m", 1.0, (M * L / (T ** 2)) * L))
        self.add(Unit("newton_meter_alt", "Nm", 1.0, (M * L / (T ** 2)) * L))
        # Temperature handled as special (affine), but keep symbol for recognition
        self.add(Unit("kelvin", "K", 1.0, Th))
        self.add(Unit("celsius", "C", 1.0, Th))
        self.add(Unit("fahrenheit", "F", 1.0, Th))
        # Amount of substance
        self.add(Unit("mole", "mol", 1.0, N))


    def add(self, unit: Unit) -> None:
        self._by_name[unit.name.lower()] = unit
        if unit.symbol:
            self._by_symbol[unit.symbol] = unit

    def get(self, key: str) -> Optional[Unit]:
        if not key:
            return None
        k = key.strip()
        u = self._by_name.get(k.lower())
        if u:
            return u
        return self._by_symbol.get(k)

    def convert(self, value: float, from_u: str, to_u: str) -> Tuple[float, str]:
        fu = self.get(from_u)
        tu = self.get(to_u)
        if not fu or not tu:
            raise ValueError("Unknown unit.")
        # Special temperature conversions (affine)
        if fu.name in ("celsius", "fahrenheit", "kelvin") or tu.name in ("celsius", "fahrenheit", "kelvin"):
            return self._convert_temperature(value, fu.name, tu.name), tu.symbol or tu.name
        if fu.dim.as_tuple() != tu.dim.as_tuple():
            raise ValueError("Incompatible dimensions for conversion.")
        si = value * fu.factor_to_si
        out = si / tu.factor_to_si
        return out, tu.symbol or tu.name

    def _convert_temperature(self, v: float, f: str, t: str) -> float:
        f = f.lower(); t = t.lower()
        # to Kelvin
        if f == "kelvin":
            k = v
        elif f == "celsius":
            k = v + 273.15
        elif f == "fahrenheit":
            k = (v - 32.0) * (5.0/9.0) + 273.15
        else:
            raise ValueError("Unknown temperature unit.")
        # from Kelvin
        if t == "kelvin":
            return k
        if t == "celsius":
            return k - 273.15
        if t == "fahrenheit":
            return (k - 273.15) * (9.0/5.0) + 32.0
        raise ValueError("Unknown temperature unit.")




# =============================================================================
# VECTOR + TENSOR ALGEBRA WITH DIMENSIONAL AWARENESS (ROBOTICS-GRADE)
# =============================================================================

@dataclass(frozen=True)
class Vec3:
    """
    3D vector with dimensional awareness.
    - values are in SI units for the associated dimension.
    - dim describes the physical dimension of each component (same for x,y,z).
    """
    x: float
    y: float
    z: float
    dim: Dimension = Dimension()

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: "Vec3") -> "Vec3":
        if self.dim.as_tuple() != other.dim.as_tuple():
            raise ValueError("Vec3 add: incompatible dimensions.")
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z, self.dim)

    def __sub__(self, other: "Vec3") -> "Vec3":
        if self.dim.as_tuple() != other.dim.as_tuple():
            raise ValueError("Vec3 sub: incompatible dimensions.")
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z, self.dim)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k, self.dim)

    def __truediv__(self, k: float) -> "Vec3":
        if k == 0:
            raise ValueError("Vec3 div: scalar cannot be zero.")
        return Vec3(self.x / k, self.y / k, self.z / k, self.dim)

    def magnitude(self) -> float:
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def direction(self) -> "Vec3":
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Vec3 normalize: magnitude is zero.")
        # direction is dimensionless
        return Vec3(self.x/mag, self.y/mag, self.z/mag, Dimension())

    def dot(self, other: "Vec3") -> Tuple[float, Dimension]:
        if self.dim.as_tuple() != other.dim.as_tuple():
            raise ValueError("Vec3 dot: incompatible dimensions.")
        return (self.x*other.x + self.y*other.y + self.z*other.z, self.dim * other.dim)

    def cross(self, other: "Vec3") -> "Vec3":
        if self.dim.as_tuple() != other.dim.as_tuple():
            raise ValueError("Vec3 cross: incompatible dimensions.")
        return Vec3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x,
            self.dim * other.dim
        )

    def project_onto(self, axis: "Vec3") -> "Vec3":
        # Projection of self onto axis: (self·axis_hat) * axis_hat
        ah = axis.direction()
        scalar, _ = Vec3(self.x, self.y, self.z, self.dim).dot(Vec3(ah.x, ah.y, ah.z, Dimension()))
        return Vec3(ah.x, ah.y, ah.z, Dimension()) * scalar  # returns same dim as self

    def format(self, unit_hint: str = "") -> str:
        u = f" {unit_hint}".rstrip()
        return f"({self.x}, {self.y}, {self.z}){u}"


@dataclass(frozen=True)
class Tensor33:
    """
    3x3 tensor with dimensional awareness (all components share same dim).
    Storage is row-major.
    """
    m00: float; m01: float; m02: float
    m10: float; m11: float; m12: float
    m20: float; m21: float; m22: float
    dim: Dimension = Dimension()

    def rows(self) -> Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]:
        return (
            (self.m00, self.m01, self.m02),
            (self.m10, self.m11, self.m12),
            (self.m20, self.m21, self.m22),
        )

    def trace(self) -> float:
        return self.m00 + self.m11 + self.m22

    def transpose(self) -> "Tensor33":
        return Tensor33(
            self.m00, self.m10, self.m20,
            self.m01, self.m11, self.m21,
            self.m02, self.m12, self.m22,
            self.dim
        )

    def det(self) -> float:
        # Laplace expansion
        a,b,c = self.m00, self.m01, self.m02
        d,e,f = self.m10, self.m11, self.m12
        g,h,i = self.m20, self.m21, self.m22
        return (
            a*(e*i - f*h)
            - b*(d*i - f*g)
            + c*(d*h - e*g)
        )

    def mul_vec(self, v: Vec3) -> Vec3:
        # Tensor-vector multiplication: dims multiply
        return Vec3(
            self.m00*v.x + self.m01*v.y + self.m02*v.z,
            self.m10*v.x + self.m11*v.y + self.m12*v.z,
            self.m20*v.x + self.m21*v.y + self.m22*v.z,
            self.dim * v.dim
        )

    def format(self, unit_hint: str = "") -> str:
        u = f" {unit_hint}".rstrip()
        r = self.rows()
        return f"[[{r[0][0]}, {r[0][1]}, {r[0][2]}], [{r[1][0]}, {r[1][1]}, {r[1][2]}], [{r[2][0]}, {r[2][1]}, {r[2][2]}]]{u}"


def _dim_to_unit_hint(dim: Dimension) -> str:
    # Not a full unit synthesis engine; provides human hints for the common robotics stack.
    d = dim.as_tuple()
    # dimensionless
    if d == Dimension().as_tuple():
        return ""
    # length
    if d == Dimension(L=1).as_tuple():
        return "m"
    # velocity
    if d == (Dimension(L=1) / Dimension(T=1)).as_tuple():
        return "m/s"
    # acceleration
    if d == (Dimension(L=1) / (Dimension(T=1) ** 2)).as_tuple():
        return "m/s^2"
    # force
    if d == (Dimension(M=1) * Dimension(L=1) / (Dimension(T=1) ** 2)).as_tuple():
        return "N"
    # torque / energy
    if d == (Dimension(M=1) * (Dimension(L=1) ** 2) / (Dimension(T=1) ** 2)).as_tuple():
        return "N*m"
    return ""


def _parse_vec3_literal(s: str) -> Optional[Tuple[float,float,float]]:
    """
    Parse (x,y,z) or [x,y,z] or <x,y,z>
    """
    if not s:
        return None
    m = re.search(r"[\(\[\<]\s*([\-0-9\.eE\+]+)\s*[, ]\s*([\-0-9\.eE\+]+)\s*[, ]\s*([\-0-9\.eE\+]+)\s*[\)\]\>]", s)
    if not m:
        return None
    try:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except Exception:
        return None


def _parse_tensor33_literal(s: str) -> Optional[Tuple[float, ...]]:
    """
    Parse a 3x3 tensor in either:
      - [[a,b,c],[d,e,f],[g,h,i]]
      - [a b c; d e f; g h i]
    Returns 9 floats row-major.
    """
    if not s:
        return None

    # Bracket form
    m = re.search(
        r"\[\s*\[\s*([^\]]+?)\s*\]\s*,\s*\[\s*([^\]]+?)\s*\]\s*,\s*\[\s*([^\]]+?)\s*\]\s*\]",
        s
    )
    if m:
        rows = []
        for grp in (m.group(1), m.group(2), m.group(3)):
            parts = [p.strip() for p in re.split(r"[, ]+", grp.strip()) if p.strip()]
            if len(parts) != 3:
                return None
            rows.append(parts)
        flat = [rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0], rows[2][1], rows[2][2]]
        try:
            return tuple(float(x) for x in flat)
        except Exception:
            return None

    # Semicolon form
    m2 = re.search(r"\[\s*([^\]]+?)\s*\]", s)
    if m2 and ";" in m2.group(1):
        raw = m2.group(1)
        row_strs = [r.strip() for r in raw.split(";") if r.strip()]
        if len(row_strs) != 3:
            return None
        rows = []
        for rs in row_strs:
            parts = [p.strip() for p in rs.replace(",", " ").split() if p.strip()]
            if len(parts) != 3:
                return None
            rows.append(parts)
        flat = [rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0], rows[2][1], rows[2][2]]
        try:
            return tuple(float(x) for x in flat)
        except Exception:
            return None

    return None


# =============================================================================
# PHYSICAL CONSTANTS + PERIODIC TABLE (HARD-CODED KNOWLEDGE)
# =============================================================================

PHYS_CONSTANTS: Dict[str, float] = {
    # Physics / thermo
    "c": 299_792_458.0,                 # speed of light (m/s)
    "h": 6.62607015e-34,                # Planck constant (J*s)
    "hbar": 1.054571817e-34,            # reduced Planck constant (J*s)
    "G": 6.67430e-11,                   # gravitational constant (N*m^2/kg^2)
    "kB": 1.380649e-23,                 # Boltzmann constant (J/K)
    "R": 8.31446261815324,              # ideal gas constant (J/mol/K)
    "NA": 6.02214076e23,                # Avogadro constant (1/mol)
    "e_charge": 1.602176634e-19,        # elementary charge (C)
    "me": 9.1093837015e-31,             # electron mass (kg)
    "mp": 1.67262192369e-27,            # proton mass (kg)
    "mn": 1.67492749804e-27,            # neutron mass (kg)
    "amu": 1.66053906660e-27,           # atomic mass unit (kg)
    "atm": 101_325.0,                   # standard atmosphere (Pa)
}

# Periodic table: atomic_number -> (symbol, name, atomic_weight)
# Source: standard atomic weights (representative values). This is a static, hard-coded table for deterministic use.
PERIODIC_TABLE: Dict[str, Dict[str, Any]] = {
    "H":  {"Z": 1,  "name": "Hydrogen",      "aw": 1.008},
    "He": {"Z": 2,  "name": "Helium",        "aw": 4.002602},
    "Li": {"Z": 3,  "name": "Lithium",       "aw": 6.94},
    "Be": {"Z": 4,  "name": "Beryllium",     "aw": 9.0121831},
    "B":  {"Z": 5,  "name": "Boron",         "aw": 10.81},
    "C":  {"Z": 6,  "name": "Carbon",        "aw": 12.011},
    "N":  {"Z": 7,  "name": "Nitrogen",      "aw": 14.007},
    "O":  {"Z": 8,  "name": "Oxygen",        "aw": 15.999},
    "F":  {"Z": 9,  "name": "Fluorine",      "aw": 18.998403163},
    "Ne": {"Z": 10, "name": "Neon",          "aw": 20.1797},
    "Na": {"Z": 11, "name": "Sodium",        "aw": 22.98976928},
    "Mg": {"Z": 12, "name": "Magnesium",     "aw": 24.305},
    "Al": {"Z": 13, "name": "Aluminium",     "aw": 26.9815385},
    "Si": {"Z": 14, "name": "Silicon",       "aw": 28.085},
    "P":  {"Z": 15, "name": "Phosphorus",    "aw": 30.973761998},
    "S":  {"Z": 16, "name": "Sulfur",        "aw": 32.06},
    "Cl": {"Z": 17, "name": "Chlorine",      "aw": 35.45},
    "Ar": {"Z": 18, "name": "Argon",         "aw": 39.948},
    "K":  {"Z": 19, "name": "Potassium",     "aw": 39.0983},
    "Ca": {"Z": 20, "name": "Calcium",       "aw": 40.078},
    "Sc": {"Z": 21, "name": "Scandium",      "aw": 44.955908},
    "Ti": {"Z": 22, "name": "Titanium",      "aw": 47.867},
    "V":  {"Z": 23, "name": "Vanadium",      "aw": 50.9415},
    "Cr": {"Z": 24, "name": "Chromium",      "aw": 51.9961},
    "Mn": {"Z": 25, "name": "Manganese",     "aw": 54.938044},
    "Fe": {"Z": 26, "name": "Iron",          "aw": 55.845},
    "Co": {"Z": 27, "name": "Cobalt",        "aw": 58.933194},
    "Ni": {"Z": 28, "name": "Nickel",        "aw": 58.6934},
    "Cu": {"Z": 29, "name": "Copper",        "aw": 63.546},
    "Zn": {"Z": 30, "name": "Zinc",          "aw": 65.38},
    "Ga": {"Z": 31, "name": "Gallium",       "aw": 69.723},
    "Ge": {"Z": 32, "name": "Germanium",     "aw": 72.630},
    "As": {"Z": 33, "name": "Arsenic",       "aw": 74.921595},
    "Se": {"Z": 34, "name": "Selenium",      "aw": 78.971},
    "Br": {"Z": 35, "name": "Bromine",       "aw": 79.904},
    "Kr": {"Z": 36, "name": "Krypton",       "aw": 83.798},
    "Rb": {"Z": 37, "name": "Rubidium",      "aw": 85.4678},
    "Sr": {"Z": 38, "name": "Strontium",     "aw": 87.62},
    "Y":  {"Z": 39, "name": "Yttrium",       "aw": 88.90584},
    "Zr": {"Z": 40, "name": "Zirconium",     "aw": 91.224},
    "Nb": {"Z": 41, "name": "Niobium",       "aw": 92.90637},
    "Mo": {"Z": 42, "name": "Molybdenum",    "aw": 95.95},
    "Tc": {"Z": 43, "name": "Technetium",    "aw": 98.0},
    "Ru": {"Z": 44, "name": "Ruthenium",     "aw": 101.07},
    "Rh": {"Z": 45, "name": "Rhodium",       "aw": 102.90550},
    "Pd": {"Z": 46, "name": "Palladium",     "aw": 106.42},
    "Ag": {"Z": 47, "name": "Silver",        "aw": 107.8682},
    "Cd": {"Z": 48, "name": "Cadmium",       "aw": 112.414},
    "In": {"Z": 49, "name": "Indium",        "aw": 114.818},
    "Sn": {"Z": 50, "name": "Tin",           "aw": 118.710},
    "Sb": {"Z": 51, "name": "Antimony",      "aw": 121.760},
    "Te": {"Z": 52, "name": "Tellurium",     "aw": 127.60},
    "I":  {"Z": 53, "name": "Iodine",        "aw": 126.90447},
    "Xe": {"Z": 54, "name": "Xenon",         "aw": 131.293},
    "Cs": {"Z": 55, "name": "Cesium",        "aw": 132.90545196},
    "Ba": {"Z": 56, "name": "Barium",        "aw": 137.327},
    "La": {"Z": 57, "name": "Lanthanum",     "aw": 138.90547},
    "Ce": {"Z": 58, "name": "Cerium",        "aw": 140.116},
    "Pr": {"Z": 59, "name": "Praseodymium",  "aw": 140.90766},
    "Nd": {"Z": 60, "name": "Neodymium",     "aw": 144.242},
    "Pm": {"Z": 61, "name": "Promethium",    "aw": 145.0},
    "Sm": {"Z": 62, "name": "Samarium",      "aw": 150.36},
    "Eu": {"Z": 63, "name": "Europium",      "aw": 151.964},
    "Gd": {"Z": 64, "name": "Gadolinium",    "aw": 157.25},
    "Tb": {"Z": 65, "name": "Terbium",       "aw": 158.92535},
    "Dy": {"Z": 66, "name": "Dysprosium",    "aw": 162.500},
    "Ho": {"Z": 67, "name": "Holmium",       "aw": 164.93033},
    "Er": {"Z": 68, "name": "Erbium",        "aw": 167.259},
    "Tm": {"Z": 69, "name": "Thulium",       "aw": 168.93422},
    "Yb": {"Z": 70, "name": "Ytterbium",     "aw": 173.045},
    "Lu": {"Z": 71, "name": "Lutetium",      "aw": 174.9668},
    "Hf": {"Z": 72, "name": "Hafnium",       "aw": 178.49},
    "Ta": {"Z": 73, "name": "Tantalum",      "aw": 180.94788},
    "W":  {"Z": 74, "name": "Tungsten",      "aw": 183.84},
    "Re": {"Z": 75, "name": "Rhenium",       "aw": 186.207},
    "Os": {"Z": 76, "name": "Osmium",        "aw": 190.23},
    "Ir": {"Z": 77, "name": "Iridium",       "aw": 192.217},
    "Pt": {"Z": 78, "name": "Platinum",      "aw": 195.084},
    "Au": {"Z": 79, "name": "Gold",          "aw": 196.966569},
    "Hg": {"Z": 80, "name": "Mercury",       "aw": 200.592},
    "Tl": {"Z": 81, "name": "Thallium",      "aw": 204.38},
    "Pb": {"Z": 82, "name": "Lead",          "aw": 207.2},
    "Bi": {"Z": 83, "name": "Bismuth",       "aw": 208.98040},
    "Po": {"Z": 84, "name": "Polonium",      "aw": 209.0},
    "At": {"Z": 85, "name": "Astatine",      "aw": 210.0},
    "Rn": {"Z": 86, "name": "Radon",         "aw": 222.0},
    "Fr": {"Z": 87, "name": "Francium",      "aw": 223.0},
    "Ra": {"Z": 88, "name": "Radium",        "aw": 226.0},
    "Ac": {"Z": 89, "name": "Actinium",      "aw": 227.0},
    "Th": {"Z": 90, "name": "Thorium",       "aw": 232.0377},
    "Pa": {"Z": 91, "name": "Protactinium",  "aw": 231.03588},
    "U":  {"Z": 92, "name": "Uranium",       "aw": 238.02891},
    "Np": {"Z": 93, "name": "Neptunium",     "aw": 237.0},
    "Pu": {"Z": 94, "name": "Plutonium",     "aw": 244.0},
    "Am": {"Z": 95, "name": "Americium",     "aw": 243.0},
    "Cm": {"Z": 96, "name": "Curium",        "aw": 247.0},
    "Bk": {"Z": 97, "name": "Berkelium",     "aw": 247.0},
    "Cf": {"Z": 98, "name": "Californium",   "aw": 251.0},
    "Es": {"Z": 99, "name": "Einsteinium",   "aw": 252.0},
    "Fm": {"Z": 100,"name": "Fermium",       "aw": 257.0},
    "Md": {"Z": 101,"name": "Mendelevium",   "aw": 258.0},
    "No": {"Z": 102,"name": "Nobelium",      "aw": 259.0},
    "Lr": {"Z": 103,"name": "Lawrencium",    "aw": 266.0},
    "Rf": {"Z": 104,"name": "Rutherfordium", "aw": 267.0},
    "Db": {"Z": 105,"name": "Dubnium",       "aw": 268.0},
    "Sg": {"Z": 106,"name": "Seaborgium",    "aw": 269.0},
    "Bh": {"Z": 107,"name": "Bohrium",       "aw": 270.0},
    "Hs": {"Z": 108,"name": "Hassium",       "aw": 269.0},
    "Mt": {"Z": 109,"name": "Meitnerium",    "aw": 278.0},
    "Ds": {"Z": 110,"name": "Darmstadtium",  "aw": 281.0},
    "Rg": {"Z": 111,"name": "Roentgenium",   "aw": 282.0},
    "Cn": {"Z": 112,"name": "Copernicium",   "aw": 285.0},
    "Nh": {"Z": 113,"name": "Nihonium",      "aw": 286.0},
    "Fl": {"Z": 114,"name": "Flerovium",     "aw": 289.0},
    "Mc": {"Z": 115,"name": "Moscovium",     "aw": 290.0},
    "Lv": {"Z": 116,"name": "Livermorium",   "aw": 293.0},
    "Ts": {"Z": 117,"name": "Tennessine",    "aw": 294.0},
    "Og": {"Z": 118,"name": "Oganesson",     "aw": 294.0},
}

def get_element(symbol_or_name: str) -> Optional[Dict[str, Any]]:
    if not symbol_or_name:
        return None
    s = symbol_or_name.strip()
    # symbol
    if s in PERIODIC_TABLE:
        return PERIODIC_TABLE[s]
    # name lookup
    sl = s.lower()
    for sym, rec in PERIODIC_TABLE.items():
        if rec.get("name", "").lower() == sl:
            return {**rec, "symbol": sym}
    return None

# =============================================================================
# FORMULA LIBRARY (HARD-CODED SCIENCE / ENGINEERING KNOWLEDGE)
# =============================================================================

@dataclass
class Formula:
    name: str
    domain: str
    equation: str
    variables: Dict[str, str]        # var -> description
    units: Dict[str, str]            # var -> unit symbol/name
    dims: Dict[str, Dimension]       # var -> dimensional vector
    solve: Callable[[Dict[str, float]], Dict[str, float]]  # returns computed vars
    notes: str = ""

class FormulaLibrary:
    """
    Hard-coded formulas + deterministic solvers.
    This is the “enterprise knowledge base” for math-driven reasoning.
    """
    def __init__(self, units: UnitRegistry) -> None:
        self.units = units
        self.formulas: Dict[str, Formula] = {}
        self._init_core()

    def register(self, f: Formula) -> None:
        self.formulas[f.name.lower()] = f

    def find(self, hint: str) -> List[Formula]:
        h = _lower(hint)
        out = []
        for k, f in self.formulas.items():
            if h in k or h in f.domain.lower() or h in f.equation.lower():
                out.append(f)
        return out

    def _init_core(self) -> None:
        # Base dims
        L = Dimension(L=1)
        M = Dimension(M=1)
        T = Dimension(T=1)
        I = Dimension(I=1)
        Th = Dimension(Th=1)

        # Derived dims
        N_dim = M * L / (T ** 2)            # newton
        J_dim = N_dim * L                   # joule
        W_dim = J_dim / T                   # watt
        Pa_dim = N_dim / (L ** 2)           # pascal
        V_dim = (M * (L ** 2)) / (T ** 3) / I
        Ohm_dim = V_dim / I

        # Newton 2nd law: F = m a
        def solve_n2(known: Dict[str, float]) -> Dict[str, float]:
            out = {}
            if "m" in known and "a" in known and "F" not in known:
                out["F"] = known["m"] * known["a"]
            if "F" in known and "m" in known and "a" not in known:
                if known["m"] == 0:
                    raise ValueError("m cannot be zero.")
                out["a"] = known["F"] / known["m"]
            if "F" in known and "a" in known and "m" not in known:
                if known["a"] == 0:
                    raise ValueError("a cannot be zero.")
                out["m"] = known["F"] / known["a"]
            return out

        self.register(Formula(
            name="Newtons Second Law",
            domain="physics.mechanics",
            equation="F = m * a",
            variables={"F": "Force", "m": "Mass", "a": "Acceleration"},
            units={"F": "N", "m": "kg", "a": "m/s^2"},
            dims={"F": N_dim, "m": M, "a": L/(T**2)},
            solve=solve_n2,
            notes="Deterministic mechanics baseline."
        ))

        # Kinetic energy: KE = 1/2 m v^2
        def solve_ke(known: Dict[str, float]) -> Dict[str, float]:
            out = {}
            if "m" in known and "v" in known and "KE" not in known:
                out["KE"] = 0.5 * known["m"] * (known["v"] ** 2)
            if "KE" in known and "m" in known and "v" not in known:
                if known["m"] == 0:
                    raise ValueError("m cannot be zero.")
                out["v"] = math.sqrt((2 * known["KE"]) / known["m"])
            if "KE" in known and "v" in known and "m" not in known:
                if known["v"] == 0:
                    raise ValueError("v cannot be zero.")
                out["m"] = (2 * known["KE"]) / (known["v"] ** 2)
            return out

        self.register(Formula(
            name="Kinetic Energy",
            domain="physics.energy",
            equation="KE = 0.5 * m * v^2",
            variables={"KE": "Kinetic Energy", "m": "Mass", "v": "Velocity"},
            units={"KE": "J", "m": "kg", "v": "m/s"},
            dims={"KE": J_dim, "m": M, "v": L/T},
            solve=solve_ke
        ))

        # Ohm's law: V = I R
        def solve_ohm(known: Dict[str, float]) -> Dict[str, float]:
            out = {}
            if "I" in known and "R" in known and "V" not in known:
                out["V"] = known["I"] * known["R"]
            if "V" in known and "R" in known and "I" not in known:
                if known["R"] == 0:
                    raise ValueError("R cannot be zero.")
                out["I"] = known["V"] / known["R"]
            if "V" in known and "I" in known and "R" not in known:
                if known["I"] == 0:
                    raise ValueError("I cannot be zero.")
                out["R"] = known["V"] / known["I"]
            return out

        self.register(Formula(
            name="Ohms Law",
            domain="ee.circuits",
            equation="V = I * R",
            variables={"V": "Voltage", "I": "Current", "R": "Resistance"},
            units={"V": "V", "I": "A", "R": "ohm"},
            dims={"V": V_dim, "I": I, "R": Ohm_dim},
            solve=solve_ohm
        ))

        # Power electrical: P = V I
        def solve_pvi(known: Dict[str, float]) -> Dict[str, float]:
            out = {}
            if "V" in known and "I" in known and "P" not in known:
                out["P"] = known["V"] * known["I"]
            if "P" in known and "V" in known and "I" not in known:
                if known["V"] == 0:
                    raise ValueError("V cannot be zero.")
                out["I"] = known["P"] / known["V"]
            if "P" in known and "I" in known and "V" not in known:
                if known["I"] == 0:
                    raise ValueError("I cannot be zero.")
                out["V"] = known["P"] / known["I"]
            return out

        self.register(Formula(
            name="Electrical Power",
            domain="ee.power",
            equation="P = V * I",
            variables={"P": "Power", "V": "Voltage", "I": "Current"},
            units={"P": "W", "V": "V", "I": "A"},
            dims={"P": W_dim, "V": V_dim, "I": I},
            solve=solve_pvi
        ))

        # Thermodynamics: Q = ΔU + W  -> ΔU = Q - W
        def solve_firstlaw(known: Dict[str, float]) -> Dict[str, float]:
            out = {}
            if "Q" in known and "W" in known and "dU" not in known:
                out["dU"] = known["Q"] - known["W"]
            if "dU" in known and "W" in known and "Q" not in known:
                out["Q"] = known["dU"] + known["W"]
            if "Q" in known and "dU" in known and "W" not in known:
                out["W"] = known["Q"] - known["dU"]
            return out

        self.register(Formula(
            name="First Law Thermodynamics",
            domain="mech.thermo",
            equation="Q = dU + W",
            variables={"Q": "Heat added", "dU": "Change in internal energy", "W": "Work done"},
            units={"Q": "J", "dU": "J", "W": "J"},
            dims={"Q": J_dim, "dU": J_dim, "W": J_dim},
            solve=solve_firstlaw
        ))

        # Fluid: Reynolds number Re = rho v D / mu
        def solve_re(known: Dict[str, float]) -> Dict[str, float]:
            out = {}
            if all(k in known for k in ("rho", "v", "D", "mu")) and "Re" not in known:
                out["Re"] = (known["rho"] * known["v"] * known["D"]) / known["mu"]
            return out

        self.register(Formula(
            name="Reynolds Number",
            domain="mech.fluids",
            equation="Re = rho * v * D / mu",
            variables={"Re": "Reynolds number", "rho": "Density", "v": "Velocity", "D": "Diameter", "mu": "Dynamic viscosity"},
            units={"Re": "dimensionless", "rho": "kg/m^3", "v": "m/s", "D": "m", "mu": "Pa*s"},
            dims={"Re": Dimension(), "rho": M/(L**3), "v": L/T, "D": L, "mu": Pa_dim*T},
            solve=solve_re
        ))

        # Ideal Gas Law: P V = n R T  (P in Pa, V in m^3, n in mol, T in K)
        def solve_ideal_gas(known: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            R = PHYS_CONSTANTS["R"]
            P = known.get("P")
            Vv = known.get("V")
            n = known.get("n")
            Tt = known.get("T")
            # Solve for any missing single variable
            if P is None and Vv is not None and n is not None and Tt is not None:
                if Vv == 0:
                    raise ValueError("V cannot be zero.")
                out["P"] = (n * R * Tt) / Vv
            elif Vv is None and P is not None and n is not None and Tt is not None:
                if P == 0:
                    raise ValueError("P cannot be zero.")
                out["V"] = (n * R * Tt) / P
            elif n is None and P is not None and Vv is not None and Tt is not None:
                if (R * Tt) == 0:
                    raise ValueError("R*T cannot be zero.")
                out["n"] = (P * Vv) / (R * Tt)
            elif Tt is None and P is not None and Vv is not None and n is not None:
                if (n * R) == 0:
                    raise ValueError("n*R cannot be zero.")
                out["T"] = (P * Vv) / (n * R)
            return out

        self.register(Formula(
            name="Ideal Gas Law",
            domain="chem.thermo",
            equation="P * V = n * R * T",
            variables={"P": "Pressure", "V": "Volume", "n": "Moles", "T": "Temperature"},
            units={"P": "Pa", "V": "m^3", "n": "mol", "T": "K"},
            dims={"P": Pa_dim, "V": L**3, "n": Dimension(N=1), "T": Th},
            solve=solve_ideal_gas,
            notes="Assumes ideal gas behavior; R is universal gas constant."
        ))

        # pH definition: pH = -log10([H+]) where [H+] in mol/L
        def solve_ph(known: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            ph = known.get("pH")
            H = known.get("H")  # hydrogen ion concentration
            if ph is None and H is not None:
                if H <= 0:
                    raise ValueError("[H+] must be > 0.")
                out["pH"] = -math.log10(H)
            elif H is None and ph is not None:
                out["H"] = 10 ** (-ph)
            return out

        self.register(Formula(
            name="pH Definition",
            domain="chem.acidbase",
            equation="pH = -log10([H+])",
            variables={"pH": "Acidity (pH)", "H": "Hydrogen ion concentration [H+]"},
            units={"pH": "dimensionless", "H": "mol/L"},
            dims={"pH": Dimension(), "H": Dimension(N=1) / (L**3)},
            solve=solve_ph,
            notes="Uses base-10 logarithm; assumes aqueous solution."
        ))

        # Mass–Energy equivalence: E = m c^2
        def solve_emc2(known: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            c = PHYS_CONSTANTS["c"]
            E = known.get("E")
            m = known.get("m")
            if E is None and m is not None:
                out["E"] = m * (c ** 2)
            elif m is None and E is not None:
                out["m"] = E / (c ** 2)
            return out

        self.register(Formula(
            name="Mass-Energy Equivalence",
            domain="physics.relativity",
            equation="E = m * c^2",
            variables={"E": "Energy", "m": "Mass"},
            units={"E": "J", "m": "kg"},
            dims={"E": J_dim, "m": M},
            solve=solve_emc2,
            notes="Relativistic rest energy."
        ))

        # Radioactive decay: N = N0 * exp(-lambda * t)
        def solve_decay(known: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            N = known.get("N")
            N0 = known.get("N0")
            lam = known.get("lambda")
            t = known.get("t")
            # compute N
            if N is None and N0 is not None and lam is not None and t is not None:
                out["N"] = N0 * math.exp(-lam * t)
            # compute lambda
            if lam is None and N is not None and N0 is not None and t is not None:
                if N <= 0 or N0 <= 0 or t == 0:
                    raise ValueError("Require N>0, N0>0, t!=0.")
                out["lambda"] = -math.log(N / N0) / t
            # compute t
            if t is None and N is not None and N0 is not None and lam is not None:
                if N <= 0 or N0 <= 0 or lam == 0:
                    raise ValueError("Require N>0, N0>0, lambda!=0.")
                out["t"] = -math.log(N / N0) / lam
            return out

        self.register(Formula(
            name="Radioactive Decay",
            domain="physics.nuclear",
            equation="N = N0 * exp(-lambda * t)",
            variables={"N": "Remaining quantity", "N0": "Initial quantity", "lambda": "Decay constant", "t": "Time"},
            units={"N": "count", "N0": "count", "lambda": "1/s", "t": "s"},
            dims={"N": Dimension(), "N0": Dimension(), "lambda": Dimension() / T, "t": T},
            solve=solve_decay,
            notes="Exponential decay law. Half-life t1/2 = ln(2)/lambda."
        ))



# =============================================================================
# LOGIC + REASONING ENGINE (SELECT FORMULA, SOLVE, EXPLAIN)
# =============================================================================

@dataclass
class SolveResult:
    ok: bool
    kind: str  # calc|convert|solve|explain|translate
    value: Any = None
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    meaning: Optional[MeaningGraph] = None

class ReasoningEngine:
    """
    The executive layer:
    - Detect intent (calc vs conversion vs formula solve vs translation)
    - Build meaning graph
    - Run deterministic solve
    - Generate rational language output
    """
    def __init__(self, units: UnitRegistry, formulas: FormulaLibrary) -> None:
        self.units = units
        self.formulas = formulas

    # -------------------------------
    # Intent detection (enterprise routing)
    # -------------------------------

    def route(self, query: str) -> SolveResult:
        q = _norm_space(query)
        ql = _lower(q)

        # 1) Translation
        if self._looks_like_translation(ql):
            return self.translate(q)

        # 2) Chemistry / nuclear / constants (domain routing before generic math)
        if self._looks_like_chemistry(ql):
            return self.chemistry(q)

        if self._looks_like_nuclear(ql):
            return self.nuclear(q)

        if self._looks_like_constants(ql):
            return self.constants(q)

        # 3) Unit conversion
        if self._looks_like_conversion(ql):
            return self.convert(q)

        # 3.5) Vector / tensor math (directional force, robotics-grade linear algebra)
        if self._looks_like_vector_math(ql):
            return self.vector_math(q)

        if self._looks_like_tensor_math(ql):
            return self.tensor_math(q)

        # 3.75) Scalar / Vector / Tensor Calculus (field operators + Jacobian/Hessian)
        if self._looks_like_calculus(ql):
            return self.calculus(q)

        # 4) Domain formula solve
        if self._looks_like_formula_solve(ql):
            return self.solve_formula(q)

        # 5) General math eval
        if self._looks_like_math(ql):
            return self.calc(q)

        # 6) Fallback: rational explanation attempt
        return self.explain(q)

    def _looks_like_translation(self, ql: str) -> bool:
        return (
            "translate" in ql
            or re.search(r"\b(en|english)\s*(to|->)\s*(es|spanish)\b", ql) is not None
            or re.search(r"\b(es|spanish)\s*(to|->)\s*(en|english)\b", ql) is not None
        )

    def _looks_like_chemistry(self, ql: str) -> bool:
        keywords = ["ph", "acid", "base", "molar", "mole", "element", "atomic weight", "atomic mass", "ideal gas", "pv=nrt", "gas law"]
        return any(k in ql for k in keywords)

    def _looks_like_nuclear(self, ql: str) -> bool:
        keywords = ["nuclear", "decay", "half-life", "halflife", "lambda", "e=mc", "emc2", "mass energy", "radioactive"]
        return any(k in ql for k in keywords)

    def _looks_like_constants(self, ql: str) -> bool:
        keywords = ["speed of light", "planck", "boltzmann", "avogadro", "gas constant", "gravitational constant", "elementary charge", "standard atmosphere"]
        return any(k in ql for k in keywords)


    def _looks_like_conversion(self, ql: str) -> bool:
        return ("convert" in ql) or (re.search(r"\bto\b", ql) and re.search(r"\d", ql))

    def _looks_like_formula_solve(self, ql: str) -> bool:
        # “use Newton” / “Ohm” / explicit equation / named domain
        keywords = ["newton", "ohm", "thermo", "reynolds", "kinetic energy", "first law", "formula", "solve for"]
        if any(k in ql for k in keywords):
            return True
        if re.search(r"[A-Za-z]+\s*=\s*[^=]+", ql):
            return True
        return False

    def _looks_like_math(self, ql: str) -> bool:
        return bool(re.search(r"[\d\+\-\*/\^\(\)]", ql))


    def _looks_like_vector_math(self, ql: str) -> bool:
        keywords = ["vector", "vec3", "vector3", "vector3d", "dot", "cross", "magnitude", "norm", "normalize",
                    "projection", "project", "directional", "torque", "moment", "r×f", "rxf", "force vector"]
        if any(k in ql for k in keywords):
            return True
        # detect explicit 3D literals like (1,2,3)
        if re.search(r"[\(\[\<]\s*\-?\d", ql) and re.search(r"[, ]\s*\-?\d", ql):
            return True
        return False

    def _looks_like_tensor_math(self, ql: str) -> bool:
        keywords = ["tensor", "matrix", "3x3", "trace", "det", "determinant", "transpose", "stress", "strain"]
        if any(k in ql for k in keywords):
            return True
        # detect matrix literal [[...],[...],[...]] or semicolon rows
        if "[[" in ql and "]]" in ql:
            return True
        if ";" in ql and "[" in ql and "]" in ql:
            return True
        return False



    def _looks_like_calculus(self, ql: str) -> bool:
        # Field operators + differential tensor/vector calculus
        keywords = [
            "grad", "gradient", "div", "divergence", "curl",
            "jacobian", "hessian", "partial", "d/dx", "d/dy", "d/dz",
            "nabla", "∇", "tensor calculus", "vector calculus", "scalar calculus"
        ]
        if any(k in ql for k in keywords):
            return True
        # shorthand patterns: grad( ... ), div( ... ), curl( ... )
        if re.search(r"\b(grad|div|curl|jacobian|hessian)\s*\(", ql):
            return True
        return False
    # -------------------------------
    # VECTOR MATH (Vec3 + directional force + torque)
    # -------------------------------

    def vector_math(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "vector"})
        q = _norm_space(query)
        ql = _lower(q)

        def _unit_after_literal(text: str, lit_span: Tuple[int,int]) -> str:
            # look right after the literal for a unit token, e.g. "(1,2,3) N"
            tail = text[lit_span[1]:]
            m = re.match(r"\s*([A-Za-zΩ°][A-Za-z0-9Ω°/\*\^\-]*)", tail)
            return (m.group(1) if m else "").strip()

        # Extract the first two vector literals, if present
        vec_matches = list(re.finditer(r"[\(\[\<]\s*[\-0-9\.eE\+]+\s*[, ]\s*[\-0-9\.eE\+]+\s*[, ]\s*[\-0-9\.eE\+]+\s*[\)\]\>]", q))
        v1 = v2 = None
        u1 = u2 = ""
        if len(vec_matches) >= 1:
            t = _parse_vec3_literal(vec_matches[0].group(0))
            if t:
                u1 = _unit_after_literal(q, vec_matches[0].span())
                v1 = t
        if len(vec_matches) >= 2:
            t = _parse_vec3_literal(vec_matches[1].group(0))
            if t:
                u2 = _unit_after_literal(q, vec_matches[1].span())
                v2 = t

        # Helper: build Vec3 with unit -> SI and dim from registry (if known)
        def _vec_from(vals: Tuple[float,float,float], unit: str) -> Vec3:
            unit = (unit or "").strip()
            if unit:
                uu = self.units.get(unit)
                if uu:
                    return Vec3(vals[0]*uu.factor_to_si, vals[1]*uu.factor_to_si, vals[2]*uu.factor_to_si, uu.dim)
            # default SI / dimensionless
            return Vec3(vals[0], vals[1], vals[2], Dimension())

        try:
            # OPERATIONS -------------------------------------------------------
            if "torque" in ql or "moment" in ql or "r×f" in ql or "rxf" in ql:
                # torque τ = r × F
                # Expect: r=(.. ) m, F=(.. ) N  (order flexible)
                # Use first vector as r, second as F by default.
                if not (v1 and v2):
                    mg.add("vector_request", [Term("raw", "concept", q)], op="torque")
                    return SolveResult(ok=False, kind="vector", text="Vector torque requires two vectors: r=(x,y,z) and F=(x,y,z).", meaning=mg)

                r_vec = _vec_from(v1, u1 or "m")
                f_vec = _vec_from(v2, u2 or "N")
                mg.add("uses_operation", [Term("torque", "concept")])
                tau = r_vec.cross(f_vec)
                unit_hint = _dim_to_unit_hint(tau.dim) or "N*m"
                mg.add("derives", [Term("tau", "concept", tau.as_tuple())], unit=unit_hint)
                return SolveResult(ok=True, kind="vector", value={"tau": tau.as_tuple(), "unit": unit_hint},
                                   text=f"Torque τ = r × F = {tau.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": tau.dim.as_tuple()})

            if "cross" in ql:
                if not (v1 and v2):
                    mg.add("vector_request", [Term("raw", "concept", q)], op="cross")
                    return SolveResult(ok=False, kind="vector", text="Cross product requires two vectors.", meaning=mg)
                a = _vec_from(v1, u1)
                b = _vec_from(v2, u2 or u1)
                mg.add("uses_operation", [Term("cross", "concept")])
                c = a.cross(b)
                unit_hint = _dim_to_unit_hint(c.dim)
                return SolveResult(ok=True, kind="vector",
                                   value={"cross": c.as_tuple(), "unit": unit_hint},
                                   text=f"a × b = {c.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": c.dim.as_tuple()})

            if "dot" in ql:
                if not (v1 and v2):
                    mg.add("vector_request", [Term("raw", "concept", q)], op="dot")
                    return SolveResult(ok=False, kind="vector", text="Dot product requires two vectors.", meaning=mg)
                a = _vec_from(v1, u1)
                b = _vec_from(v2, u2 or u1)
                mg.add("uses_operation", [Term("dot", "concept")])
                d, dim = a.dot(b)
                unit_hint = _dim_to_unit_hint(dim)
                return SolveResult(ok=True, kind="vector",
                                   value={"dot": d, "unit": unit_hint},
                                   text=f"a · b = {d} {unit_hint}".rstrip(),
                                   meaning=mg,
                                   meta={"dim": dim.as_tuple()})

            if "magnitude" in ql or "norm" in ql:
                if not v1:
                    mg.add("vector_request", [Term("raw", "concept", q)], op="magnitude")
                    return SolveResult(ok=False, kind="vector", text="Magnitude requires one vector literal like (x,y,z).", meaning=mg)
                a = _vec_from(v1, u1)
                mg.add("uses_operation", [Term("magnitude", "concept")])
                mag = a.magnitude()
                unit_hint = _dim_to_unit_hint(a.dim)
                return SolveResult(ok=True, kind="vector",
                                   value={"magnitude": mag, "unit": unit_hint},
                                   text=f"|v| = {mag} {unit_hint}".rstrip(),
                                   meaning=mg,
                                   meta={"dim": a.dim.as_tuple()})

            if "normalize" in ql or "unit vector" in ql or "direction" in ql:
                if not v1:
                    mg.add("vector_request", [Term("raw", "concept", q)], op="normalize")
                    return SolveResult(ok=False, kind="vector", text="Normalize requires one vector literal like (x,y,z).", meaning=mg)
                a = _vec_from(v1, u1)
                mg.add("uses_operation", [Term("normalize", "concept")])
                ah = a.direction()
                return SolveResult(ok=True, kind="vector",
                                   value={"unit_vector": ah.as_tuple()},
                                   text=f"v̂ = {ah.format()} (dimensionless)",
                                   meaning=mg,
                                   meta={"dim": ah.dim.as_tuple()})

            if "directional" in ql and ("force" in ql or "f=" in ql):
                # Build force vector from magnitude + direction: F = |F| * d̂
                # Example: "force 10 N direction (1,1,0)"
                m_mag = re.search(r"(?:force|f)\s*(?:=)?\s*(\-?\d+(?:\.\d+)?)\s*([A-Za-zΩ°][A-Za-z0-9Ω°/\*\^\-]*)?", ql)
                if not m_mag or not v1:
                    mg.add("vector_request", [Term("raw", "concept", q)], op="directional_force")
                    return SolveResult(ok=False, kind="vector", text="Directional force requires a magnitude + direction vector. Example: 'force 10 N direction (1,1,0)'.", meaning=mg)
                mag = float(m_mag.group(1))
                unit = (m_mag.group(2) or "N").strip()
                uu = self.units.get(unit) or self.units.get("N")
                if not uu:
                    raise ValueError("Unknown force unit.")
                d = _vec_from(v1, "")  # direction should be dimensionless
                dh = Vec3(d.x, d.y, d.z, Dimension()).direction()
                F = Vec3(dh.x, dh.y, dh.z, Dimension()) * (mag * uu.factor_to_si)
                F = Vec3(F.x, F.y, F.z, uu.dim)
                unit_hint = _dim_to_unit_hint(F.dim) or (uu.symbol or uu.name)
                return SolveResult(ok=True, kind="vector",
                                   value={"F": F.as_tuple(), "unit": unit_hint},
                                   text=f"F = |F|·d̂ = {F.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": F.dim.as_tuple()})

            # Default: treat as a vector literal echo with unit/dim normalization
            if v1:
                a = _vec_from(v1, u1)
                unit_hint = _dim_to_unit_hint(a.dim) or u1
                mg.add("vector", [Term("v", "concept", a.as_tuple())], unit=unit_hint)
                return SolveResult(ok=True, kind="vector",
                                   value={"v": a.as_tuple(), "unit": unit_hint},
                                   text=f"Vector3D normalized: {a.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": a.dim.as_tuple()})

            mg.add("vector_request", [Term("raw", "concept", q)])
            return SolveResult(ok=False, kind="vector", text="Vector engine ready. Provide an operation (dot, cross, magnitude, normalize, torque) and vector literals like (x,y,z).", meaning=mg)

        except Exception as e:
            return SolveResult(ok=False, kind="vector", text=f"Vector failure: {e}", meaning=mg)


    # -------------------------------
    # TENSOR MATH (3x3, stress/strain primitives)
    # -------------------------------

    def tensor_math(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "tensor"})
        q = _norm_space(query)
        ql = _lower(q)

        # Parse tensor literal
        tvals = _parse_tensor33_literal(q)
        if not tvals:
            mg.add("tensor_request", [Term("raw", "concept", q)])
            return SolveResult(ok=False, kind="tensor", text="Tensor engine expects a 3x3 literal like [[a,b,c],[d,e,f],[g,h,i]].", meaning=mg)

        # Optional unit token at end: "... ]] Pa"
        unit = ""
        m_u = re.search(r"\]\s*([A-Za-zΩ°][A-Za-z0-9Ω°/\*\^\-]*)\s*$", q)
        if m_u:
            unit = (m_u.group(1) or "").strip()

        # Build tensor with unit -> SI and dim
        uu = self.units.get(unit) if unit else None
        if uu:
            t = Tensor33(
                tvals[0]*uu.factor_to_si, tvals[1]*uu.factor_to_si, tvals[2]*uu.factor_to_si,
                tvals[3]*uu.factor_to_si, tvals[4]*uu.factor_to_si, tvals[5]*uu.factor_to_si,
                tvals[6]*uu.factor_to_si, tvals[7]*uu.factor_to_si, tvals[8]*uu.factor_to_si,
                uu.dim
            )
        else:
            t = Tensor33(*tvals, dim=Dimension())

        try:
            if "trace" in ql:
                tr = t.trace()
                unit_hint = _dim_to_unit_hint(t.dim) or unit
                return SolveResult(ok=True, kind="tensor",
                                   value={"trace": tr, "unit": unit_hint},
                                   text=f"tr(T) = {tr} {unit_hint}".rstrip(),
                                   meaning=mg,
                                   meta={"dim": t.dim.as_tuple()})

            if "det" in ql:
                detv = t.det()
                unit_hint = _dim_to_unit_hint(t.dim) or unit
                # determinant is cubic in units (dim^3) if dim not dimensionless; we report hint only
                return SolveResult(ok=True, kind="tensor",
                                   value={"det": detv, "unit_hint": unit_hint},
                                   text=f"det(T) = {detv} (unit-hint: {unit_hint}^3)".rstrip(),
                                   meaning=mg,
                                   meta={"dim": t.dim.as_tuple()})

            if "transpose" in ql:
                tt = t.transpose()
                unit_hint = _dim_to_unit_hint(tt.dim) or unit
                return SolveResult(ok=True, kind="tensor",
                                   value={"transpose": tt.rows(), "unit": unit_hint},
                                   text=f"Tᵀ = {tt.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": tt.dim.as_tuple()})

            if "mul" in ql or "apply" in ql or "·v" in ql or "tv" in ql:
                v = _parse_vec3_literal(q)
                if not v:
                    mg.add("tensor_request", [Term("raw", "concept", q)], op="mul_vec")
                    return SolveResult(ok=False, kind="tensor", text="Tensor·vector requires a vector literal (x,y,z) in the query.", meaning=mg)
                vv = Vec3(v[0], v[1], v[2], Dimension())
                outv = t.mul_vec(vv)
                unit_hint = _dim_to_unit_hint(outv.dim) or unit
                return SolveResult(ok=True, kind="tensor",
                                   value={"T_mul_v": outv.as_tuple(), "unit": unit_hint},
                                   text=f"T·v = {outv.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": outv.dim.as_tuple()})

            # Default: echo normalized tensor
            unit_hint = _dim_to_unit_hint(t.dim) or unit
            return SolveResult(ok=True, kind="tensor",
                               value={"T": t.rows(), "unit": unit_hint},
                               text=f"Tensor33 normalized: {t.format(unit_hint)}",
                               meaning=mg,
                               meta={"dim": t.dim.as_tuple()})

        except Exception as e:
            return SolveResult(ok=False, kind="tensor", text=f"Tensor failure: {e}", meaning=mg)


    
    # -------------------------------
    # SCALAR / VECTOR / TENSOR CALCULUS (finite-difference, unit-aware)
    # -------------------------------

    def calculus(self, query: str) -> SolveResult:
        """
        Deterministic calculus operators for scalar/vector/tensor fields.
        Supported (all numeric via central finite differences):
          - grad(f) / gradient(f)
          - div(Fx,Fy,Fz) / divergence(...)
          - curl(Fx,Fy,Fz)
          - jacobian(Fx,Fy,Fz)  (3x3)
          - hessian(f)          (3x3)
          - tensor_div(Txx,Txy,Txz,Tyx,Tyy,Tyz,Tzx,Tzy,Tzz) -> Vec3
        Usage patterns:
          grad(x*y + z) x=1 y=2 z=3 h=1e-5 unit=Pa
          div(x, y, z) x=1 y=2 z=3
          curl(y, -x, 0) x=1 y=0 z=0
          jacobian(x*y, y*z, z*x) x=1 y=2 z=3
          hessian(x*x + y*y + z*z) x=1 y=2 z=3
          tensor_div( x,0,0, 0,y,0, 0,0,z ) x=1 y=2 z=3
        """
        mg = MeaningGraph(meta={"intent": "calculus"})
        q = _norm_space(query)
        ql = _lower(q)

        # Extract evaluation point + step size
        vars_known = self._extract_assignments(q)
        x0 = float(vars_known.get("x", 0.0))
        y0 = float(vars_known.get("y", 0.0))
        z0 = float(vars_known.get("z", 0.0))
        h = float(vars_known.get("h", 1e-5))

        # Optional unit hint for the field values
        unit = ""
        m_unit = re.search(r"\b(?:unit|u)\s*=\s*([A-Za-zΩ°][A-Za-z0-9Ω°/\*\^\-]*)\b", q, flags=re.I)
        if m_unit:
            unit = (m_unit.group(1) or "").strip()

        uu = self.units.get(unit) if unit else None
        base_dim = uu.dim if uu else Dimension()
        L = Dimension(L=1)

        # Identify operator + body
        op = None
        body = None
        m = re.search(r"\b(grad|gradient|divergence|div|curl|jacobian|hessian|tensor_div|tensordiv)\s*\((.*)\)", ql)
        if m:
            op = m.group(1)
            # slice original query for body to preserve case/symbols
            # best-effort: find the first '(' after op
            p = q.lower().find(m.group(1))  # op start
            p_paren = q.find("(", p)
            if p_paren != -1:
                # find matching ')'
                depth = 0
                end = -1
                for i in range(p_paren, len(q)):
                    if q[i] == "(":
                        depth += 1
                    elif q[i] == ")":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end != -1:
                    body = q[p_paren+1:end].strip()
        else:
            # Alternate pattern: "grad f=..." or "gradient f=..."
            m2 = re.search(r"\b(grad|gradient|divergence|div|curl|jacobian|hessian|tensor_div|tensordiv)\b\s*(.+)", ql)
            if m2:
                op = m2.group(1)
                body = q[len(m2.group(1)):].strip()

        if not op or not body:
            mg.add("calculus_request", [Term("raw", "concept", q)])
            return SolveResult(ok=False, kind="calculus",
                               text="Calculus engine expects an operator like grad(...), div(...), curl(...), jacobian(...), hessian(...), tensor_div(...), plus x=,y=,z= and optional h=.",
                               meaning=mg)

        op_norm = op.lower()
        if op_norm == "gradient":
            op_norm = "grad"
        if op_norm == "divergence":
            op_norm = "div"
        if op_norm == "tensordiv":
            op_norm = "tensor_div"

        # Safe evaluation for expressions in x,y,z
        def eval_expr(expr: str, x: float, y: float, z: float) -> float:
            return self._safe_eval_xyz(expr, x=x, y=y, z=z)

        def d_dx(expr: str, x: float, y: float, z: float) -> float:
            return (eval_expr(expr, x+h, y, z) - eval_expr(expr, x-h, y, z)) / (2.0*h)

        def d_dy(expr: str, x: float, y: float, z: float) -> float:
            return (eval_expr(expr, x, y+h, z) - eval_expr(expr, x, y-h, z)) / (2.0*h)

        def d_dz(expr: str, x: float, y: float, z: float) -> float:
            return (eval_expr(expr, x, y, z+h) - eval_expr(expr, x, y, z-h)) / (2.0*h)

        # Parse arguments
        args = self._split_top_level_args(body)

        try:
            if op_norm == "grad":
                if len(args) != 1:
                    return SolveResult(ok=False, kind="calculus", text="grad(f) requires exactly one scalar expression f(x,y,z).", meaning=mg)
                f = args[0]
                gx = d_dx(f, x0, y0, z0)
                gy = d_dy(f, x0, y0, z0)
                gz = d_dz(f, x0, y0, z0)
                dim_out = base_dim / L
                v = Vec3(gx, gy, gz, dim_out)
                unit_hint = _dim_to_unit_hint(dim_out)
                mg.add("uses_operation", [Term("grad", "concept")])
                return SolveResult(ok=True, kind="calculus",
                                   value={"grad": v.as_tuple(), "unit": unit_hint, "at": {"x": x0, "y": y0, "z": z0}, "h": h},
                                   text=f"∇f = {v.format(unit_hint)} at (x,y,z)=({x0},{y0},{z0})",
                                   meaning=mg,
                                   meta={"dim": dim_out.as_tuple()})

            if op_norm == "div":
                if len(args) != 3:
                    return SolveResult(ok=False, kind="calculus", text="div(Fx,Fy,Fz) requires three component expressions.", meaning=mg)
                Fx, Fy, Fz = args[0], args[1], args[2]
                divv = d_dx(Fx, x0, y0, z0) + d_dy(Fy, x0, y0, z0) + d_dz(Fz, x0, y0, z0)
                dim_out = base_dim / L
                unit_hint = _dim_to_unit_hint(dim_out)
                mg.add("uses_operation", [Term("div", "concept")])
                return SolveResult(ok=True, kind="calculus",
                                   value={"div": divv, "unit": unit_hint, "at": {"x": x0, "y": y0, "z": z0}, "h": h},
                                   text=f"∇·F = {divv} {unit_hint}".rstrip(),
                                   meaning=mg,
                                   meta={"dim": dim_out.as_tuple()})

            if op_norm == "curl":
                if len(args) != 3:
                    return SolveResult(ok=False, kind="calculus", text="curl(Fx,Fy,Fz) requires three component expressions.", meaning=mg)
                Fx, Fy, Fz = args[0], args[1], args[2]
                cx = d_dy(Fz, x0, y0, z0) - d_dz(Fy, x0, y0, z0)
                cy = d_dz(Fx, x0, y0, z0) - d_dx(Fz, x0, y0, z0)
                cz = d_dx(Fy, x0, y0, z0) - d_dy(Fx, x0, y0, z0)
                dim_out = base_dim / L
                v = Vec3(cx, cy, cz, dim_out)
                unit_hint = _dim_to_unit_hint(dim_out)
                mg.add("uses_operation", [Term("curl", "concept")])
                return SolveResult(ok=True, kind="calculus",
                                   value={"curl": v.as_tuple(), "unit": unit_hint, "at": {"x": x0, "y": y0, "z": z0}, "h": h},
                                   text=f"∇×F = {v.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": dim_out.as_tuple()})

            if op_norm == "jacobian":
                if len(args) != 3:
                    return SolveResult(ok=False, kind="calculus", text="jacobian(Fx,Fy,Fz) requires three component expressions.", meaning=mg)
                Fx, Fy, Fz = args[0], args[1], args[2]
                j00 = d_dx(Fx, x0, y0, z0); j01 = d_dy(Fx, x0, y0, z0); j02 = d_dz(Fx, x0, y0, z0)
                j10 = d_dx(Fy, x0, y0, z0); j11 = d_dy(Fy, x0, y0, z0); j12 = d_dz(Fy, x0, y0, z0)
                j20 = d_dx(Fz, x0, y0, z0); j21 = d_dy(Fz, x0, y0, z0); j22 = d_dz(Fz, x0, y0, z0)
                dim_out = base_dim / L
                T = Tensor33(j00, j01, j02, j10, j11, j12, j20, j21, j22, dim_out)
                unit_hint = _dim_to_unit_hint(dim_out)
                mg.add("uses_operation", [Term("jacobian", "concept")])
                return SolveResult(ok=True, kind="calculus",
                                   value={"jacobian": T.rows(), "unit": unit_hint, "at": {"x": x0, "y": y0, "z": z0}, "h": h},
                                   text=f"J = {T.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": dim_out.as_tuple()})

            if op_norm == "hessian":
                if len(args) != 1:
                    return SolveResult(ok=False, kind="calculus", text="hessian(f) requires exactly one scalar expression.", meaning=mg)
                f = args[0]
                # second derivatives via central differences on first derivatives (stable enough for deterministic baseline)
                f_x = lambda X,Y,Z: d_dx(f, X, Y, Z)
                f_y = lambda X,Y,Z: d_dy(f, X, Y, Z)
                f_z = lambda X,Y,Z: d_dz(f, X, Y, Z)

                h00 = (f_x(x0+h, y0, z0) - f_x(x0-h, y0, z0)) / (2.0*h)
                h11 = (f_y(x0, y0+h, z0) - f_y(x0, y0-h, z0)) / (2.0*h)
                h22 = (f_z(x0, y0, z0+h) - f_z(x0, y0, z0-h)) / (2.0*h)

                h01 = (f_x(x0, y0+h, z0) - f_x(x0, y0-h, z0)) / (2.0*h)
                h02 = (f_x(x0, y0, z0+h) - f_x(x0, y0, z0-h)) / (2.0*h)
                h10 = (f_y(x0+h, y0, z0) - f_y(x0-h, y0, z0)) / (2.0*h)
                h12 = (f_y(x0, y0, z0+h) - f_y(x0, y0, z0-h)) / (2.0*h)
                h20 = (f_z(x0+h, y0, z0) - f_z(x0-h, y0, z0)) / (2.0*h)
                h21 = (f_z(x0, y0+h, z0) - f_z(x0, y0-h, z0)) / (2.0*h)

                dim_out = base_dim / (L ** 2)
                H = Tensor33(h00, h01, h02, h10, h11, h12, h20, h21, h22, dim_out)
                unit_hint = _dim_to_unit_hint(dim_out)
                mg.add("uses_operation", [Term("hessian", "concept")])
                return SolveResult(ok=True, kind="calculus",
                                   value={"hessian": H.rows(), "unit": unit_hint, "at": {"x": x0, "y": y0, "z": z0}, "h": h},
                                   text=f"H = {H.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": dim_out.as_tuple()})

            if op_norm == "tensor_div":
                if len(args) != 9:
                    return SolveResult(ok=False, kind="calculus", text="tensor_div(Txx,Txy,Txz,Tyx,Tyy,Tyz,Tzx,Tzy,Tzz) requires 9 component expressions.", meaning=mg)
                Txx,Txy,Txz,Tyx,Tyy,Tyz,Tzx,Tzy,Tzz = args
                # (∇·T)_i = ∂T_{ij}/∂x_j  with j in {x,y,z}
                vx = d_dx(Txx, x0,y0,z0) + d_dy(Txy, x0,y0,z0) + d_dz(Txz, x0,y0,z0)
                vy = d_dx(Tyx, x0,y0,z0) + d_dy(Tyy, x0,y0,z0) + d_dz(Tyz, x0,y0,z0)
                vz = d_dx(Tzx, x0,y0,z0) + d_dy(Tzy, x0,y0,z0) + d_dz(Tzz, x0,y0,z0)
                dim_out = base_dim / L
                v = Vec3(vx, vy, vz, dim_out)
                unit_hint = _dim_to_unit_hint(dim_out)
                mg.add("uses_operation", [Term("tensor_div", "concept")])
                return SolveResult(ok=True, kind="calculus",
                                   value={"tensor_div": v.as_tuple(), "unit": unit_hint, "at": {"x": x0, "y": y0, "z": z0}, "h": h},
                                   text=f"∇·T = {v.format(unit_hint)}",
                                   meaning=mg,
                                   meta={"dim": dim_out.as_tuple()})

            return SolveResult(ok=False, kind="calculus", text=f"Unknown calculus operator: {op_norm}", meaning=mg)

        except Exception as e:
            return SolveResult(ok=False, kind="calculus", text=f"Calculus failure: {e}", meaning=mg)

    def _safe_eval_xyz(self, expr: str, x: float, y: float, z: float) -> float:
        """
        Safe numeric evaluation of expressions in x,y,z.
        Deterministic, restricted builtins.
        """
        if not expr:
            raise ValueError("Empty expression.")
        e = expr.strip()
        e = e.replace("^", "**").replace("×", "*").replace("÷", "/")
        allowed = {
            "x": float(x), "y": float(y), "z": float(z),
            "pi": math.pi, "e": math.e, "tau": math.tau,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "abs": abs, "floor": math.floor, "ceil": math.ceil,
            "pow": pow, "min": min, "max": max,
        }
        if not re.fullmatch(r"[0-9\.\+\-\*/\^\(\)\sA-Za-z_]+", e):
            raise ValueError("Unsafe expression.")
        return float(eval(e, {"__builtins__": {}}, allowed))

    def _split_top_level_args(self, s: str) -> List[str]:
        """
        Split a comma-separated argument list, respecting (),[],{} nesting.
        """
        out: List[str] = []
        buf: List[str] = []
        depth = 0
        for ch in (s or ""):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    out.append(part)
                buf = []
            else:
                buf.append(ch)
        last = "".join(buf).strip()
        if last:
            out.append(last)
        return out
# -------------------------------
    # CALC (safe eval baseline)
    # -------------------------------

    def calc(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "calc"})
        expr = self._normalize_math_expr(query)
        mg.add("evaluates", [Term("expression", "concept", expr)])

        try:
            # Minimal safe eval: digits + operators + math funcs/constants
            allowed = {"pi": math.pi, "e": math.e, "tau": math.tau, "sqrt": math.sqrt,
                       "sin": math.sin, "cos": math.cos, "tan": math.tan,
                       "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
                       "floor": math.floor, "ceil": math.ceil, "pow": pow}
            if not re.fullmatch(r"[0-9\.\+\-\*/\^\(\)\sA-Za-z_]+", expr):
                raise ValueError("Unsafe expression.")
            expr = expr.replace("^", "**")
            val = eval(expr, {"__builtins__": {}}, allowed)  # controlled environment
            return SolveResult(ok=True, kind="calc", value=val, text=self._nlg_calc(expr, val), meaning=mg,
                               meta={"expression": expr})
        except Exception as e:
            return SolveResult(ok=False, kind="calc", value=None, text=f"Calc failure: {e}", meaning=mg)

    def _normalize_math_expr(self, q: str) -> str:
        q = q.strip()
        q = q.replace("×", "*").replace("÷", "/")
        q = re.sub(r"\bplus\b", "+", q, flags=re.I)
        q = re.sub(r"\bminus\b", "-", q, flags=re.I)
        q = re.sub(r"\btimes\b", "*", q, flags=re.I)
        q = re.sub(r"\bdivided by\b", "/", q, flags=re.I)
        q = _norm_space(q)
        return q

    # -------------------------------
    # CONVERT
    # -------------------------------

    def convert(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "convert"})
        # patterns:
        # "convert 10 km to m"
        # "10 km to m"
        ql = _lower(query)

        m = re.search(r"(?:convert\s*)?(\-?\d+(?:\.\d+)?)\s*([A-Za-zΩ°]+)\s*(?:to|in)\s*([A-Za-zΩ°]+)", ql)
        if not m:
            mg.add("convert_request", [Term("raw", "concept", query)])
            return SolveResult(ok=False, kind="convert", text="Conversion parse failure.", meaning=mg)

        val = float(m.group(1))
        from_u = m.group(2)
        to_u = m.group(3)

        mg.add("converts_to", [Term("value", "number", val), Term(from_u, "unit"), Term(to_u, "unit")])

        try:
            out, out_sym = self.units.convert(val, from_u, to_u)
            return SolveResult(
                ok=True,
                kind="convert",
                value=out,
                text=self._nlg_convert(val, from_u, out, out_sym),
                meaning=mg,
                meta={"from": from_u, "to": to_u}
            )
        except Exception as e:
            return SolveResult(ok=False, kind="convert", text=f"Conversion failure: {e}", meaning=mg)

    

    # -------------------------------
    # CHEMISTRY (acid/base, gas law, periodic table)
    # -------------------------------

    def chemistry(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "chemistry"})
        q = _norm_space(query)
        ql = _lower(q)

        # 1) Periodic table lookups
        m_el = re.search(r"(?:element|atomic weight of|atomic mass of|molar mass of)\s+([A-Za-z]{1,2}|[A-Za-z]+)", ql)
        if m_el:
            token = m_el.group(1).strip()
            # Try symbol first (case sensitive symbol set), then name
            rec = get_element(token.capitalize()) or get_element(token)
            if rec:
                sym = token.capitalize() if token.capitalize() in PERIODIC_TABLE else rec.get("symbol", token)
                mg.add("element_lookup", [Term(sym, "entity"), Term("atomic_weight", "concept", rec.get("aw"))], Z=rec.get("Z"))
                aw = rec.get("aw")
                return SolveResult(ok=True, kind="chemistry", value=rec,
                                   text=f"Element {rec.get('name')} ({sym}), Z={rec.get('Z')}, atomic weight≈{aw} g/mol",
                                   meaning=mg)

        # 2) pH queries: "pH of 1e-3" or "pH when H=0.001"
        if "ph" in ql:
            known = self._extract_assignments(q)
            # allow patterns like "pH of 1e-3" meaning H=1e-3
            m = re.search(r"ph\s*(?:of|for)?\s*([0-9\.eE\-\+]+)", ql)
            if m and "H" not in known and "pH" not in known:
                try:
                    known["H"] = float(m.group(1))
                except Exception:
                    pass
            # normalize variable names
            if "h" in {k.lower() for k in known.keys()} and "H" not in known:
                for k,v in list(known.items()):
                    if k.lower()=="h":
                        known["H"]=v; del known[k]
                        break
            # Use formula library pH
            f = self.formulas.formulas.get("ph definition")
            if f:
                mg.add("uses_formula", [Term(f.name, "concept"), Term(f.equation, "concept")], domain=f.domain)
                mg.add("has_knowns", [Term(k, "symbol", v) for k, v in known.items()])
                try:
                    computed = f.solve(known)
                    if computed:
                        mg.add("derives", [Term(k, "symbol", v) for k, v in computed.items()])
                        if "pH" in computed:
                            return SolveResult(ok=True, kind="chemistry", value=computed, text=f"pH ≈ {computed['pH']}", meaning=mg)
                        if "H" in computed:
                            return SolveResult(ok=True, kind="chemistry", value=computed, text=f"[H+] ≈ {computed['H']} mol/L", meaning=mg)
                except Exception as e:
                    return SolveResult(ok=False, kind="chemistry", text=f"Chemistry failure: {e}", meaning=mg)

        # 3) Ideal gas law solver: accepts assignments P,V,n,T
        if "ideal gas" in ql or "pv=nrt" in ql or "gas law" in ql:
            known = self._extract_assignments(q)
            # Common aliases
            alias = {"p": "P", "v": "V", "t": "T", "n": "n"}
            for k, v in list(known.items()):
                kl = k.lower()
                if kl in alias and k != alias[kl]:
                    known[alias[kl]] = known.pop(k)
            f = self.formulas.formulas.get("ideal gas law")
            if f:
                mg.add("uses_formula", [Term(f.name, "concept"), Term(f.equation, "concept")], domain=f.domain)
                mg.add("has_knowns", [Term(k, "symbol", v) for k, v in known.items()])
                try:
                    computed = f.solve(known)
                    if computed:
                        mg.add("derives", [Term(k, "symbol", v) for k, v in computed.items()])
                        # One-line summary
                        parts = [f"{k}={v}" for k, v in computed.items()]
                        return SolveResult(ok=True, kind="chemistry", value=computed,
                                           text="Ideal gas solve -> " + ", ".join(parts),
                                           meaning=mg)
                    return SolveResult(ok=False, kind="chemistry", text="Insufficient known variables for ideal gas solve (need 3 of P,V,n,T).", meaning=mg)
                except Exception as e:
                    return SolveResult(ok=False, kind="chemistry", text=f"Chemistry failure: {e}", meaning=mg)

        # Fallback
        mg.add("chemistry_request", [Term("raw", "concept", q)])
        return SolveResult(ok=True, kind="chemistry", value=None,
                           text="Chemistry engine ready. Provide explicit variables (e.g., P=..., V=..., n=..., T=... or pH with [H+]).",
                           meaning=mg)

    # -------------------------------
    # NUCLEAR / ATOMIC (decay, E=mc^2)
    # -------------------------------

    def nuclear(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "nuclear"})
        q = _norm_space(query)
        ql = _lower(q)

        # Half-life convenience: lambda = ln(2)/t_half
        if "half" in ql and "life" in ql:
            known = self._extract_assignments(q)
            # accept t_half or thalf
            t_half = known.get("t_half") or known.get("thalf") or known.get("t1/2")
            if t_half is not None:
                lam = math.log(2.0) / t_half
                mg.add("derives", [Term("lambda", "symbol", lam)], rule="lambda=ln2/t_half")
                return SolveResult(ok=True, kind="nuclear", value={"lambda": lam},
                                   text=f"Decay constant λ ≈ {lam} 1/s (from t_half={t_half}s)",
                                   meaning=mg)

        # Radioactive decay formula
        if "decay" in ql or "lambda" in ql:
            known = self._extract_assignments(q)
            f = self.formulas.formulas.get("radioactive decay")
            if f:
                mg.add("uses_formula", [Term(f.name, "concept"), Term(f.equation, "concept")], domain=f.domain)
                mg.add("has_knowns", [Term(k, "symbol", v) for k, v in known.items()])
                try:
                    computed = f.solve(known)
                    if computed:
                        mg.add("derives", [Term(k, "symbol", v) for k, v in computed.items()])
                        return SolveResult(ok=True, kind="nuclear", value=computed,
                                           text="Decay solve -> " + ", ".join(f"{k}={v}" for k, v in computed.items()),
                                           meaning=mg)
                    return SolveResult(ok=False, kind="nuclear", text="Insufficient known variables for decay solve (need 3 of N,N0,lambda,t).", meaning=mg)
                except Exception as e:
                    return SolveResult(ok=False, kind="nuclear", text=f"Nuclear failure: {e}", meaning=mg)

        # E=mc^2
        if "e=mc" in ql or "emc2" in ql or "mass energy" in ql:
            known = self._extract_assignments(q)
            f = self.formulas.formulas.get("mass-energy equivalence")
            if f:
                mg.add("uses_formula", [Term(f.name, "concept"), Term(f.equation, "concept")], domain=f.domain)
                mg.add("has_knowns", [Term(k, "symbol", v) for k, v in known.items()])
                try:
                    computed = f.solve(known)
                    if computed:
                        mg.add("derives", [Term(k, "symbol", v) for k, v in computed.items()])
                        return SolveResult(ok=True, kind="nuclear", value=computed,
                                           text="E=mc^2 -> " + ", ".join(f"{k}={v}" for k, v in computed.items()),
                                           meaning=mg)
                except Exception as e:
                    return SolveResult(ok=False, kind="nuclear", text=f"Nuclear failure: {e}", meaning=mg)

        mg.add("nuclear_request", [Term("raw", "concept", q)])
        return SolveResult(ok=True, kind="nuclear", value=None,
                           text="Nuclear engine ready. Provide explicit variables (e.g., m=..., E=..., or N0=..., lambda=..., t=...).",
                           meaning=mg)

    # -------------------------------
    # CONSTANTS (deterministic factual retrieval)
    # -------------------------------

    def constants(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "constants"})
        q = _norm_space(query)
        ql = _lower(q)

        # map phrases to keys
        mapping = {
            "speed of light": "c",
            "planck": "h",
            "reduced planck": "hbar",
            "gravitational constant": "G",
            "boltzmann": "kB",
            "avogadro": "NA",
            "gas constant": "R",
            "elementary charge": "e_charge",
            "standard atmosphere": "atm",
        }
        key = None
        for phrase, k in mapping.items():
            if phrase in ql:
                key = k
                break

        if key and key in PHYS_CONSTANTS:
            val = PHYS_CONSTANTS[key]
            mg.add("constant", [Term(key, "symbol", val)])
            return SolveResult(ok=True, kind="constants", value={"key": key, "value": val},
                               text=f"Constant {key} = {val}",
                               meaning=mg)

        mg.add("constants_request", [Term("raw", "concept", q)])
        return SolveResult(ok=True, kind="constants", value=None,
                           text="Constants engine ready. Ask for constants like: speed of light, Planck constant, Avogadro constant, gas constant, Boltzmann constant.",
                           meaning=mg)

# -------------------------------
    # FORMULA SOLVE (math-driven reasoning)
    # -------------------------------

    def solve_formula(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "solve_formula"})
        ql = _lower(query)

        # 1) Identify formula candidate by keyword
        candidates: List[Formula] = []
        if "ohm" in ql:
            candidates += self.formulas.find("ohms law")
        if "newton" in ql:
            candidates += self.formulas.find("newtons second law")
        if "reynolds" in ql:
            candidates += self.formulas.find("reynolds")
        if "kinetic" in ql:
            candidates += self.formulas.find("kinetic energy")
        if "first law" in ql or "thermo" in ql:
            candidates += self.formulas.find("first law thermodynamics")

        # 2) If explicit equation present, try match by equation token
        eq_match = re.search(r"([A-Za-z]+)\s*=\s*([^=]+)", query)
        if eq_match and not candidates:
            eq = _norm_space(eq_match.group(0))
            candidates = [f for f in self.formulas.formulas.values() if _lower(f.equation) == _lower(eq)]

        if not candidates:
            mg.add("no_formula_match", [Term("query", "concept", query)])
            return SolveResult(ok=False, kind="solve", text="No formula matched.", meaning=mg)

        f = candidates[0]
        mg.add("uses_formula", [Term(f.name, "concept"), Term(f.equation, "concept")], domain=f.domain)

        # 3) Extract known variables (pattern: m=2, a=3, V=5, I=2, etc.)
        known = self._extract_assignments(query)
        mg.add("has_knowns", [Term(k, "symbol", v) for k, v in known.items()])

        try:
            computed = f.solve(known)
            if not computed:
                return SolveResult(ok=False, kind="solve", text="Insufficient known variables to solve.", meaning=mg,
                                   meta={"formula": f.name, "required": list(f.variables.keys()), "known": known})

            # 4) Build rational explanation
            mg.add("derives", [Term(k, "symbol", v) for k, v in computed.items()])
            text = self._nlg_solve_formula(f, known, computed)
            return SolveResult(ok=True, kind="solve", value=computed, text=text, meaning=mg,
                               meta={"formula": f.name, "equation": f.equation, "known": known, "computed": computed})
        except Exception as e:
            return SolveResult(ok=False, kind="solve", text=f"Solve failure: {e}", meaning=mg,
                               meta={"formula": f.name, "known": known})

    def _extract_assignments(self, query: str) -> Dict[str, float]:
        # Accept formats:
        # m=2, a=3
        # m = 2 kg
        # V=5, I=2
        out: Dict[str, float] = {}
        for m in re.finditer(r"\b([A-Za-z]{1,4})\s*=\s*(\-?\d+(?:\.\d+)?)", query):
            out[m.group(1)] = float(m.group(2))
        return out

    # -------------------------------
    # EXPLAIN (math/logic to language)
    # -------------------------------

    def explain(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "explain"})
        mg.add("requests_explanation", [Term("query", "concept", query)])

        # Deterministic “logic framing”:
        # Convert statement to structured claim, then generate rational reply.
        claims = self._extract_claims(query)
        for c in claims:
            mg.add("claim", [Term("text", "concept", c)])

        text = self._nlg_explain(query, claims)
        return SolveResult(ok=True, kind="explain", value={"claims": claims}, text=text, meaning=mg)

    def _extract_claims(self, query: str) -> List[str]:
        # Minimal claim segmentation, expandable
        parts = re.split(r"[.;\n]+", query)
        return [p.strip() for p in parts if len(p.strip()) > 0]

    # -------------------------------
    # TRANSLATE (interlingua-driven “Rosetta Stone”)
    # -------------------------------

    def translate(self, query: str) -> SolveResult:
        mg = MeaningGraph(meta={"intent": "translate"})

        # Supported patterns:
        # "translate: <text> en->es"
        # "translate <text> to spanish"
        ql = _lower(query)

        src, dst = self._parse_lang_pair(ql)
        payload = self._parse_translate_payload(query)

        if not payload:
            mg.add("translate_request", [Term("raw", "concept", query)], src=src, dst=dst)
            return SolveResult(ok=False, kind="translate", text="Translation parse failure.", meaning=mg)

        # Build meaning graph via lightweight semantics
        # (Tokenize -> map to concepts -> assemble propositions)
        inter = self._interlingua_from_text(payload, src=src)
        mg.props.extend(inter.props)
        mg.meta.update({"src": src, "dst": dst})

        # Generate target language from interlingua using language packs
        out_text = self._generate_from_interlingua(inter, dst=dst)

        return SolveResult(ok=True, kind="translate", value=out_text, text=out_text, meaning=mg,
                           meta={"src": src, "dst": dst, "input": payload})

    def _parse_lang_pair(self, ql: str) -> Tuple[str, str]:
        # default: English -> Spanish as baseline demonstration
        src, dst = "en", "es"
        m = re.search(r"\b(en|english)\s*(to|->)\s*(es|spanish)\b", ql)
        if m:
            return ("en", "es")
        m = re.search(r"\b(es|spanish)\s*(to|->)\s*(en|english)\b", ql)
        if m:
            return ("es", "en")
        if "to spanish" in ql:
            return ("en", "es")
        if "to english" in ql:
            return ("es", "en")
        return (src, dst)

    def _parse_translate_payload(self, query: str) -> str:
        # Remove leading directive
        q = query.strip()
        q = re.sub(r"^\s*translate\s*[:\-]?\s*", "", q, flags=re.I)
        # Strip trailing lang hint
        q = re.sub(r"\s*\b(en|english)\s*(to|->)\s*(es|spanish)\b\s*$", "", q, flags=re.I)
        q = re.sub(r"\s*\b(es|spanish)\s*(to|->)\s*(en|english)\b\s*$", "", q, flags=re.I)
        q = re.sub(r"\s*\bto\s+(spanish|english)\b\s*$", "", q, flags=re.I)
        return _norm_space(q)

    # -------------------------------
    # Interlingua model (math/logic semantics)
    # -------------------------------

    def _interlingua_from_text(self, text: str, src: str = "en") -> MeaningGraph:
        """
        Lightweight interlingua:
        - Tokenize
        - Map lexemes -> concepts
        - Detect intent (request/command/assertion)
        - Assemble propositions

        This is the scalable enterprise approach: semantics first, language second.
        """
        mg = MeaningGraph(meta={"src": src, "type": "interlingua"})
        t = _norm_space(text)
        tl = _lower(t)

        # Intent typing
        intent = "assertion"
        if tl.endswith("?") or any(w in tl.split()[:3] for w in ("what", "why", "how", "when", "where", "who")):
            intent = "question"
        if any(tl.startswith(w) for w in ("please", "do", "make", "create", "open", "close", "calculate")):
            intent = "command"
        mg.meta["intent"] = intent

        # Basic concept extraction
        concepts = self._map_concepts(tl, src=src)
        mg.add("has_text", [Term("text", "concept", t)])
        for c in concepts:
            mg.add("mentions", [Term(c, "concept")])

        # Basic predicate assembly (expand over time)
        if intent == "question":
            mg.add("asks_about", [Term("topic", "concept", concepts[0] if concepts else "unknown")])
        elif intent == "command":
            mg.add("requests_action", [Term("action", "concept", concepts[0] if concepts else "unknown")])
        else:
            mg.add("states", [Term("statement", "concept", t)])

        return mg

    def _map_concepts(self, tl: str, src: str) -> List[str]:
        # Minimal bilingual lexicon map; this is where “everything possible” scales:
        # you expand concept tables per domain (physics, fluids, EE, etc.) and per language.
        lex = LanguagePacks.get(src).lexicon
        concepts: List[str] = []
        for token in re.findall(r"[a-zA-ZáéíóúñüΩ]+", tl):
            c = lex.get(token)
            if c and c not in concepts:
                concepts.append(c)
        return concepts

    def _generate_from_interlingua(self, mg: MeaningGraph, dst: str) -> str:
        pack = LanguagePacks.get(dst)
        # Priority: preserve direct original for unknown content, but translate recognized concepts.
        # Enterprise behavior: graceful degradation; never hallucinate.
        original = ""
        for p in mg.props:
            if p.predicate == "has_text":
                original = str(p.args[0].value or "")
                break

        if not original:
            return pack.render_fallback(mg)

        # Token-level deterministic translation (baseline), enhanced by concept-level rendering
        # 1) Try concept-level sentence templates if intent is recognized
        intent = mg.meta.get("intent", "assertion")
        mentions = [p.args[0].name for p in mg.props if p.predicate == "mentions" and p.args]
        if mentions:
            templ = pack.templates.get(intent)
            if templ:
                topic = pack.concept_to_word(mentions[0]) or mentions[0]
                return templ.format(topic=topic)

        # 2) Fallback: word-by-word deterministic map for known lexemes
        return pack.word_translate(original)

    # -------------------------------
    # NLG (Rational replies)
    # -------------------------------

    def _nlg_calc(self, expr: str, val: Any) -> str:
        return f"Computed result using deterministic evaluation: {expr} = {val}"

    def _nlg_convert(self, v: float, from_u: str, out: float, out_u: str) -> str:
        return f"Unit conversion executed with dimensional controls: {v} {from_u} = {out} {out_u}"

    def _nlg_solve_formula(self, f: Formula, known: Dict[str, float], computed: Dict[str, float]) -> str:
        # Enterprise explanation with substitution + output
        lines = []
        lines.append(f"Formula selected: {f.name} ({f.domain})")
        lines.append(f"Equation: {f.equation}")
        if known:
            lines.append("Known inputs:")
            for k, v in known.items():
                u = f.units.get(k, "")
                lines.append(f" - {k} = {v} {u}".rstrip())
        lines.append("Derived outputs:")
        for k, v in computed.items():
            u = f.units.get(k, "")
            lines.append(f" - {k} = {v} {u}".rstrip())
        lines.append("Rationale: deterministic algebraic isolation and substitution.")
        return "\n".join(lines)

    def _nlg_explain(self, query: str, claims: List[str]) -> str:
        # This is the “math-to-language” baseline: structured response, no hype.
        lines = []
        lines.append("Rational response generated from structured claims.")
        for i, c in enumerate(claims, 1):
            lines.append(f"{i}. {c}")
        lines.append("Next step: provide variables/constraints to formalize and solve deterministically.")
        return "\n".join(lines)


# =============================================================================
# LANGUAGE PACKS (TRANSLATION CONTROL PLANE)
# =============================================================================

class LanguagePack:
    """
    Deterministic language pack:
    - lexicon maps word -> concept
    - reverse maps concept -> word
    - templates render interlingua into language
    - word_translate provides graceful degradation
    """
    def __init__(self, code: str) -> None:
        self.code = code
        self.lexicon: Dict[str, str] = {}          # word -> concept
        self.reverse: Dict[str, str] = {}          # concept -> preferred word
        self.templates: Dict[str, str] = {}        # intent -> template

    def add(self, word: str, concept: str, preferred: Optional[str] = None) -> None:
        self.lexicon[word.lower()] = concept
        if preferred:
            self.reverse[concept] = preferred

    def concept_to_word(self, concept: str) -> Optional[str]:
        return self.reverse.get(concept)

    def word_translate(self, text: str) -> str:
        # Deterministic token translation: preserves punctuation/unknowns
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        out: List[str] = []
        for tok in tokens:
            low = tok.lower()
            if low in self.lexicon:
                concept = self.lexicon[low]
                out.append(self.reverse.get(concept, tok))
            else:
                out.append(tok)
        # Restore spacing reasonably
        s = ""
        for i, tok in enumerate(out):
            if i == 0:
                s = tok
            elif re.fullmatch(r"[.,;:!?)]", tok):
                s += tok
            elif re.fullmatch(r"[(]", tok):
                s += " " + tok
            else:
                s += " " + tok
        return s

    def render_fallback(self, mg: MeaningGraph) -> str:
        # Enterprise fallback: no hallucination
        return "Translation engine: insufficient semantic coverage for deterministic render."


class LanguagePacks:
    _packs: Dict[str, LanguagePack] = {}

    @classmethod
    def bootstrap(cls) -> None:
        # English pack
        en = LanguagePack("en")
        en.templates = {
            "question": "You are asking about: {topic}. Provide variables/constraints for a deterministic solution.",
            "command": "Request recognized: {topic}. Provide parameters to execute deterministically.",
            "assertion": "Statement acknowledged about: {topic}."
        }

        # Spanish pack
        es = LanguagePack("es")
        es.templates = {
            "question": "Estás preguntando sobre: {topic}. Proporciona variables/restricciones para una solución determinista.",
            "command": "Solicitud reconocida: {topic}. Proporciona parámetros para ejecutar de forma determinista.",
            "assertion": "Declaración reconocida sobre: {topic}."

        }

        # French pack
        fr = LanguagePack("fr")
        fr.templates = {
            "question": "Vous demandez à propos de : {topic}. Fournissez des variables/contraintes pour une solution déterministe.",
            "command": "Demande reconnue : {topic}. Fournissez des paramètres pour exécuter de façon déterministe.",
            "assertion": "Déclaration reconnue à propos de : {topic}."
        }

        # German pack
        de = LanguagePack("de")
        de.templates = {
            "question": "Du fragst nach: {topic}. Bitte Variablen/Randbedingungen für eine deterministische Lösung angeben.",
            "command": "Anfrage erkannt: {topic}. Bitte Parameter angeben, um deterministisch auszuführen.",
            "assertion": "Aussage erkannt über: {topic}."
        }

        # Core concepts (expand aggressively)
        # Concepts are language-agnostic keys: "concept.physics.force", etc.
        # English terms
        en.add("force", "concept.physics.force", preferred="force")
        en.add("mass", "concept.physics.mass", preferred="mass")
        en.add("acceleration", "concept.physics.acceleration", preferred="acceleration")
        en.add("voltage", "concept.ee.voltage", preferred="voltage")
        en.add("current", "concept.ee.current", preferred="current")
        en.add("resistance", "concept.ee.resistance", preferred="resistance")
        en.add("energy", "concept.physics.energy", preferred="energy")
        en.add("pressure", "concept.fluids.pressure", preferred="pressure")

        # Spanish terms mapped to same concepts
        es.add("fuerza", "concept.physics.force", preferred="fuerza")
        es.add("masa", "concept.physics.mass", preferred="masa")
        es.add("aceleración", "concept.physics.acceleration", preferred="aceleración")
        es.add("voltaje", "concept.ee.voltage", preferred="voltaje")
        es.add("corriente", "concept.ee.current", preferred="corriente")
        es.add("resistencia", "concept.ee.resistance", preferred="resistencia")
        es.add("energía", "concept.physics.energy", preferred="energía")
        es.add("presión", "concept.fluids.pressure", preferred="presión")

        cls._packs["en"] = en
        cls._packs["es"] = es
        cls._packs["fr"] = fr
        cls._packs["de"] = de

    @classmethod
    def get(cls, code: str) -> LanguagePack:
        c = (code or "en").lower()
        if not cls._packs:
            cls.bootstrap()
        return cls._packs.get(c, cls._packs["en"])


# =============================================================================
# PUBLIC FACADE (THIS IS WHAT OTHER CORE FILES CALL)
# =============================================================================

class SarahMemoryLogicCalc:
    """
    Single entrypoint for AiOS:
    - route(query) -> SolveResult (structured)
    - provides meaning graph for downstream “rational language” and governance
    """
    def __init__(self) -> None:
        self.units = UnitRegistry()
        self.formulas = FormulaLibrary(self.units)
        self.engine = ReasoningEngine(self.units, self.formulas)

    def route(self, query: str) -> Dict[str, Any]:
        start = _now_ms()
        res = self.engine.route(query)
        dur = _now_ms() - start
        payload = {
            "ok": res.ok,
            "kind": res.kind,
            "value": res.value,
            "text": res.text,
            "meta": {**res.meta, "duration_ms": dur},
            "meaning": res.meaning.summarize() if res.meaning else None
        }
        return payload


# Global instance (import-safe)
LogicCalc = SarahMemoryLogicCalc()
# ====================================================================
# END OF SarahMemoryLogicCalc.py v8.0.0
# ====================================================================