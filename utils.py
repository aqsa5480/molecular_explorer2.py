"""
Utility functions for Molecular Explorer Pro.

- Local chemical lookup
- Formula/name -> SMILES conversion (cached)
- PubChem metadata fetch
- RDKit molecule preparation (3D if possible, 2D fallback)
- Element parsing utilities
- Quantum support filter (for EMT demo calc)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Set

import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core.composition import Composition  # used in other parts of app


# ---------------------------------------------------------------------------
# Local reference database (fast lookup; displayed in sidebar)
# ---------------------------------------------------------------------------
CHEMICAL_DATABASE: Dict[str, str] = {
    "Methane (CH4)": "CH4", "Ethane (C2H6)": "C2H6",
    "Propane (C3H8)": "C3H8", "Butane (C4H10)": "C4H10",
    "Methanol (CH3OH)": "CH3OH", "Ethanol (C2H5OH)": "C2H5OH",
    "Formic Acid (HCOOH)": "HCOOH", "Acetic Acid (CH3COOH)": "CH3COOH",
    "Water (H2O)": "H2O", "Ammonia (NH3)": "NH3",
    "Carbon Dioxide (CO2)": "CO2", "Hydrogen (H2)": "H2",
    "Sodium Hydroxide (NaOH)": "NaOH", "Potassium Hydroxide (KOH)": "KOH",
    "Chlorine (Cl2)": "Cl2", "Fluorine (F2)": "F2",
    "Polyethylene ((C2H4)n)": "(C2H4)n", "Polystyrene ((C8H8)n)": "(C8H8)n",
    "Caffeine (C8H10N4O2)": "C8H10N4O2", "ATP (C10H16N5O13P3)": "C10H16N5O13P3",
    "Buckminsterfullerene (C60)": "C60", "Carbon Nanotube (C)": "C",
    "Glucose (C6H12O6)": "C6H12O6", "Aspirin (C9H8O4)": "C9H8O4",
    "Sodium Chloride (NaCl)": "NaCl", "Sulfuric Acid (H2SO4)": "H2SO4",
    "Hydrochloric Acid (HCl)": "HCl", "Nitric Acid (HNO3)": "HNO3",
    "Phosphoric Acid (H3PO4)": "H3PO4", "Urea (CH4N2O)": "CH4N2O",
    "Paracetamol (C8H9NO2)": "C8H9NO2", "Ibuprofen (C13H18O2)": "C13H18O2",
}


# ---------------------------------------------------------------------------
# Quick SMILES lookup table (manual overrides when PubChem is slow/ambiguous)
# ---------------------------------------------------------------------------
_SMILES_MAP: Dict[str, str] = {
    "H2O": "O",
    "CH4": "C",
    "CH3COOH": "CC(=O)O",
    "C2H5OH": "CCO",
    "CO2": "O=C=O",
    "NH3": "N",
    "NaOH": "[Na+].[OH-]",
    "KOH": "[K+].[OH-]",
    "HNO3": "O[N+](=O)[O-]",
    "HCl": "Cl",
    "H2SO4": "O=S(=O)(O)O",
    "H3PO4": "OP(=O)(O)O",
    "NaCl": "[Na+].[Cl-]",
    "CH4N2O": "NC(=O)N",  # Urea
    "C6H12O6": "OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",  # Glucose
    "C8H9NO2": "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
    "C13H18O2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "C9H8O4": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "C8H10N4O2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "C10H16N5O13P3": "Nc1ncnc2c1ncn2[C@H]3O[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O",  # ATP
    "C60": "C1=CC2=CC3=CC4=CC5=CC6=CC7=CC8=CC9=CC1=C2C3=C4C5=C6C7=C8C9",  # Approx fullerene
    "C2H4": "C=C",
    "C8H8": "C1=CC=C(C=C1)C=C",
    "CH3OH": "CO",
    "C2H6": "CC",
    "C3H8": "CCC",
    "C4H10": "CCCC",
    "HCOOH": "O=CO",
    "NH4OH": "[NH4+].[OH-]",
    "O2": "O=O",
    "N2": "N#N",
    "Cl2": "ClCl",
    "F2": "FF",
}


# ---------------------------------------------------------------------------
# Formula / name --> SMILES
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def formula_to_smiles(query: str) -> Optional[str]:
    """
    Convert a chemical formula or common name to a SMILES string.
    Checks a local SMILES map first, then queries PubChem by name, then by formula.
    """
    q = (query or "").strip().lower()

    # Local SMILES map lookup
    if q in _SMILES_MAP:
        return _SMILES_MAP[q]

    # Check in chemical database
    for name, formula in CHEMICAL_DATABASE.items():
        if q == name.lower() or q == formula.lower():
            return _SMILES_MAP.get(formula)

    # Fallback to PubChemPy search
    try:
        comps = pcp.get_compounds(q, "name")
        if not comps:
            comps = pcp.get_compounds(q, "formula")
        if comps and comps[0].canonical_smiles:
            return comps[0].canonical_smiles
    except Exception as e:
        st.warning(f"SMILES conversion failed for '{query}': {e}")

    print(f"[Warning] SMILES not found for query: '{query}'")
    return None

# ---------------------------------------------------------------------------
# PubChem accessors
# ---------------------------------------------------------------------------
def get_pubchem_compounds(formula: str) -> List[pcp.Compound]:
    """Fetch compounds from PubChem by molecular formula."""
    try:
        return pcp.get_compounds(formula, "formula")
    except Exception as e:
        st.warning(f"PubChem server error: {str(e)}")
        return []


def get_compound_info(formula: str) -> Dict[str, str]:
    """
    Look up a compound by simple formula in local DB first, then PubChem.
    Returns a dict with display-safe fields.
    """
    for name, f in CHEMICAL_DATABASE.items():
        if f == formula:
            return {"name": name, "formula": formula}

    try:
        cmpds = get_pubchem_compounds(formula)
        if cmpds:
            c = cmpds[0]
            return {
                "name": c.iupac_name or "Unknown",
                "synonyms": ", ".join(c.synonyms[:3]) if c.synonyms else "",
                "formula": c.molecular_formula,
                "pubchem_id": c.cid,
            }
    except Exception as e:
        st.warning(f"Compound lookup failed: {str(e)}")

    return {"name": "Unknown", "formula": formula}


# ---------------------------------------------------------------------------
# RDKit molecule prep
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def prepare_mol_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDG()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"3D embedding failed: {e}")
        return None
    return mol


# ---------------------------------------------------------------------------
# Element parsing helpers
# ---------------------------------------------------------------------------
def extract_elements(formula: str) -> Set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula))


SUPPORTED_EMT: Set[str] = {"H", "C", "N", "O", "Al", "Na"}


def is_quantum_supported(formula: str) -> bool:
    """Return True if all elements in formula are in the SUPPORTED_EMT set."""
    elements = extract_elements(formula)
    return all(e in SUPPORTED_EMT for e in elements)


__all__ = [
    "CHEMICAL_DATABASE",
    "formula_to_smiles",
    "get_pubchem_compounds",
    "get_compound_info",
    "prepare_molecule",
    "extract_elements",
    "is_quantum_supported",
    "SUPPORTED_EMT",
]
missing = set(CHEMICAL_DATABASE.values()) - set(_SMILES_MAP.keys())
print("Missing SMILES for:", missing)
