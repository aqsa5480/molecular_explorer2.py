# --- Core Imports ---
import os
import re
import sys
import base64
import datetime
import time
from io import BytesIO
from typing import Dict, Optional, List, Union
from collections import Counter

# Streamlit & UI
import streamlit as st
from streamlit.components.v1 import html as st_html

# Scientific Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Chemistry Libraries
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from pymatgen.core.composition import Composition 
from ase import Atoms 
from ase.calculators.emt import EMT 
from ase.optimize import BFGS 
import pubchempy as pcp
import py3Dmol
import speech_recognition as sr
import pyttsx3
import plotly.graph_objects as go
import plotly.express as px
import requests
# Local modules
sys.path.append(".")
import sascorer
# --------------------------------
# ASE (optional – cloud safe)
# --------------------------------
ASE_ENABLED = False

try:
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    ASE_ENABLED = True
except Exception:
    ASE_ENABLED = False

# Load secret API key
api_key = st.secrets["OPENROUTER_API_KEY"]

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []


# --------------------------------
# Application configuration
# --------------------------------
st.set_page_config(
    page_title="Molecular Explorer Pro | by Aqsa Ijaz",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

import random

# --------------------------------
# Voice assistant (safe for cloud)
# --------------------------------
VOICE_ENABLED = False

try:
    import speech_recognition as sr
    import pyttsx3
    from helper import speak_text
    VOICE_ENABLED = True
except Exception:
    VOICE_ENABLED = False

# Sidebar toggle (local-only voice)
st.sidebar.markdown("### 🎙️ Voice Assistant")
voice_toggle = st.sidebar.checkbox("Enable voice (local only)", value=False)

if voice_toggle and VOICE_ENABLED:
    st.session_state["voice_allowed"] = True
else:
    st.session_state["voice_allowed"] = False

# --------------------------------
# Greeting logic
# --------------------------------
GREETING_MESSAGES = [
    "Heyy! You're back! Let's GOOOO!",
    "Whoa! Molecules await—let’s unravel their secrets!",
    "Hello scientist! You bring the curiosity, I’ll bring the chemistry!",
    "Let’s make atoms dance! Time to decode some structures!",
    "Welcome back! I’ve got a flask full of surprises today!"
]

def play_greeting():
    if not VOICE_ENABLED:
        return
    if not st.session_state.get("voice_allowed", False):
        return
    greeting = random.choice(GREETING_MESSAGES)
    speak_text(greeting, voice_type="male", rate="+45%")

# --------------------------------
# Play greeting only once per session
# --------------------------------
if "greeted" not in st.session_state:
    st.session_state.greeted = False

if not st.session_state.greeted:
    play_greeting()
    st.session_state.greeted = True

# --- Imports ---
from typing import Optional, List
import streamlit as st
from rdkit import Chem
import pubchempy as pcp
from pymatgen.core.composition import Composition

# --- SMILES dictionary ---
smiles_map = {
    "H2O": "O", "H2": "[H][H]", "O2": "O=O", "N2": "N#N", "CO2": "O=C=O",
    "NH3": "N", "CH4": "C", "C2H6": "CC", "C3H8": "CCC", "C4H10": "CCCC",
    "C2H4": "C=C", "C2H2": "C#C",
    "CH3OH": "CO", "C2H5OH": "CCO", "CH3COOH": "CC(=O)O", "HCOOH": "O=CO",
    "HCl": "Cl", "HNO3": "O[N+](=O)[O-]", "H2SO4": "O=S(=O)(O)O", "H3PO4": "OP(=O)(O)O",
    "NaOH": "[Na+].[OH-]", "KOH": "[K+].[OH-]", "NH4OH": "[NH4+].[OH-]",
    "NaCl": "[Na+].[Cl-]", "KCl": "[K+].[Cl-]", "CaCl2": "[Ca+2].[Cl-].[Cl-]",
    "C6H12O6": "OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
    "C12H22O11": "OC[C@H]1O[C@@H](O[C@H]2[C@@H](O)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H](O)[C@H]1O",
    "C8H9NO2": "CC(=O)NC1=CC=C(C=C1)O", "C13H18O2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "C9H8O4": "CC(=O)OC1=CC=CC=C1C(=O)O", "C8H10N4O2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C10H16N5O13P3": "Nc1ncnc2c1ncn2[C@H]3O[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O",
    "Cl2": "ClCl", "F2": "FF", "Br2": "BrBr", "I2": "I[I]",
}
# --- Manual names mapping ---
manual_names = {
    "H2O": "Water (H2O)", "H2": "Hydrogen (H2)", "O2": "Oxygen (O2)", "N2": "Nitrogen (N2)",
    "CO2": "Carbon Dioxide (CO2)", "NH3": "Ammonia (NH3)", "CH4": "Methane (CH4)",
    "C2H6": "Ethane (C2H6)", "C3H8": "Propane (C3H8)", "C4H10": "Butane (C4H10)",
    "C2H4": "Ethylene (C2H4)", "C2H2": "Acetylene (C2H2)",
    "CH3OH": "Methanol (CH3OH)", "C2H5OH": "Ethanol (C2H5OH)",
    "CH3COOH": "Acetic Acid (CH3COOH)", "HCOOH": "Formic Acid (HCOOH)",
    "HCl": "Hydrochloric Acid (HCl)", "HNO3": "Nitric Acid (HNO3)",
    "H2SO4": "Sulfuric Acid (H2SO4)", "H3PO4": "Phosphoric Acid (H3PO4)",
    "NaOH": "Sodium Hydroxide (NaOH)", "KOH": "Potassium Hydroxide (KOH)",
    "NH4OH": "Ammonium Hydroxide (NH4OH)",
    "NaCl": "Sodium Chloride (NaCl)", "KCl": "Potassium Chloride (KCl)",
    "CaCl2": "Calcium Chloride (CaCl2)",
}

# --- CHEMICAL_DATABASE ---
CHEMICAL_DATABASE = {}
for compound, smiles in smiles_map.items():
    name = manual_names.get(compound, compound)
    CHEMICAL_DATABASE[name] = compound
# --- PubChem fetch function ---
def get_pubchem_compounds(formula: str) -> List[pcp.Compound]:
    try:
        return pcp.get_compounds(formula, 'formula')
    except Exception as e:
        st.warning(f"PubChem server error: {str(e)}")
        return []

# --- Main formula_to_smiles function ---
@st.cache_data(show_spinner=False)
def formula_to_smiles(query: str) -> Optional[str]:
    query = query.strip()
    
    # 1️⃣ Check local SMILES map
    if query in smiles_map:
        return smiles_map[query]
    
    # 2️⃣ PubChem fallback
    try:
        compounds = pcp.get_compounds(query, 'name')
        if compounds:
            return compounds[0].canonical_smiles
        compounds = pcp.get_compounds(query, 'formula')
        if compounds:
            return compounds[0].canonical_smiles
    except Exception as e:
        st.warning(f"PubChem fetch failed: {e}")
        return None
    
    return None

# --- Formula safe function for pymatgen ---
def name_to_formula_safe(name_or_formula: str) -> str:
    """Convert chemical name to formula if known, else return as-is."""
    return smiles_map.get(name_or_formula, name_or_formula)

# --- Usage in calculation ---
def calculate_properties(user_input: str):
    formula = name_to_formula_safe(user_input)
    comp = Composition(formula)  # now safe
    return {
        "formula": str(comp),
        "mass": comp.weight,
        "elements": list(comp.get_el_amt_dict().keys())
    }

def prepare_molecule(smiles: str, embed_seed: int = 42) -> Optional[Chem.Mol]:
    """Return a molecule with Hs and valid conformer (3D or 2D)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        # Try 3D conformer generation
        res = AllChem.EmbedMolecule(mol, randomSeed=embed_seed)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            # Fallback to 2D
            AllChem.Compute2DCoords(mol)
        return mol
    except Exception as e:
        st.warning(f"Molecule preparation error: {str(e)}")
        return None
def get_compound_info(formula: str) -> Dict[str, str]:
    """Fetch comprehensive compound data with improved fallback"""
    try:
        # First try our database
        for name, f in CHEMICAL_DATABASE.items():
            if f == formula:
                return {'name': name, 'formula': formula}
        # Then try PubChem
        cmpds = get_pubchem_compounds(formula)
        if cmpds:
            c = cmpds[0]
            return {
                'name': c.iupac_name or 'Unknown',
                'synonyms': ', '.join(c.synonyms[:3]) if c.synonyms else '',
                'formula': c.molecular_formula,
                'pubchem_id': c.cid
            }
        return {'name': 'Unknown', 'formula': formula}
    except Exception as e:
        st.warning(f"Compound lookup failed: {str(e)}")
        return {'name': 'Unknown', 'formula': formula}

from typing import Optional
def speak_response(response):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        male_voice = [v for v in voices if "male" in v.name.lower()]
        engine.setProperty('voice', male_voice[0].id if male_voice else voices[0].id)
        engine.say(response)
        engine.runAndWait()
    except:
        pass
def handle_chem_query(query):
    match_formula = re.search(r"(?:formula|symbol|molecular formula) of ([\w\s\-]+)", query, re.I)
    match_props = re.search(r"(?:properties|structure|weight) of ([\w\s\-]+)", query, re.I)
    name = None
    if match_formula:
        name = match_formula.group(1).strip()
    elif match_props:
        name = match_props.group(1).strip()

    if name:
        try:
            comp = pcp.get_compounds(name, 'name')[0]
            info = []
            if comp.molecular_formula:
                info.append(f"Formula: {comp.molecular_formula}")
            if comp.molecular_weight:
                info.append(f"Mol. Weight: {comp.molecular_weight:.2f} g/mol")
            if comp.iupac_name:
                info.append(f"IUPAC: {comp.iupac_name}")
            return " | ".join(info)
        except:
            return f"❌ Couldn’t find anything about **{name}**."

    # Fallback to OpenRouter
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openrouter/auto",
            "messages": [
                {"role": "system", "content": "You are a smart chemistry tutor. Keep it brief and correct."},
                {"role": "user", "content": query}
            ]
        }
        r = requests.post(url, headers=headers, json=data)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ OpenRouter Error {r.status_code}: {r.json().get('error', {}).get('message', 'Unknown')}"
    except Exception as e:
        return f"❌ Exception: {e}"
from rdkit.Chem import Descriptors

def evaluate_lipinski(mol):
    """Evaluate Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    rules = {
        "molecular_weight < 500": mw < 500,
        "logP < 5": logp < 5,
        "hydrogen_bond_donors ≤ 5": hbd <= 5,
        "hydrogen_bond_acceptors ≤ 10": hba <= 10,
    }

    passed = all(rules.values())
    rules["passed"] = passed
    return rules


def estimate_bioavailability(mol):
    """Simple heuristic: start at 1.0 and subtract penalties."""
    hbd       = Descriptors.NumHDonors(mol)
    hba       = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa      = Descriptors.TPSA(mol)
    logp      = Descriptors.MolLogP(mol)
    score = 1.0
    if hbd > 5 or hba > 10:
        score -= 0.3
    if tpsa > 140:
        score -= 0.3
    if rot_bonds > 10:
        score -= 0.2
    if logp < -1 or logp > 5:
        score -= 0.2
    return max(score, 0.0)

def calculate_sascore(mol):
    """Wrapper around your sascorer.calculateScore call."""
    from sascorer import calculateScore
    return round(calculateScore(mol), 2)

def get_render_style(style: str, colorscheme: str):
    """
    Map user-selected style + colorscheme to a py3Dmol style dict.
    Notes:
      - 'colorscheme' works reliably for stick/sphere.
      - Cartoon on small molecules does nothing; we still return cartoon, but
        caller can layer a Stick fallback.
      - Surface color left None so py3Dmol uses atomic colors under translucent shell.
    """
    if style == "Stick":
        return {"stick": {"radius": 0.25, "colorscheme": colorscheme}}
    elif style == "Sphere":
        return {"sphere": {"scale": 0.3, "colorscheme": colorscheme}}
    elif style == "Cartoon":
        return {"cartoon": {"color": "spectrum"}}   # ignore colorscheme
    elif style == "Surface":
        return {"surface": {"opacity": 0.9}}        # let atom colors bleed through
    else:
        return {"stick": {"colorscheme": colorscheme}}


    query = query.strip()

    if query in smiles_map:
        return smiles_map[query]

    try:
        compounds = pcp.get_compounds(query, 'name')
        if not compounds:
            compounds = pcp.get_compounds(query, 'formula')
        if compounds and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
    except Exception as e:
        st.warning(f"SMILES conversion failed for '{query}': {e}")

    return None

SUPPORTED_EMT = {"H", "C", "N", "O", "Al", "Na"}

def extract_elements(formula: str) -> set:
    return set(re.findall(r"[A-Z][a-z]?", formula))

def is_quantum_supported(formula: str) -> bool:
    elements = extract_elements(formula)
    return all(e in SUPPORTED_EMT for e in elements)
def show_basic_properties(formula: str):
    st.markdown("### Molecular Formula Summary")

    # Element breakdown
    import re
    from collections import Counter

    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    element_counts = Counter()

    for elem, count in elements:
        count = int(count) if count else 1
        element_counts[elem] += count

    st.write("**Elemental Composition:**")
    for elem, count in element_counts.items():
        st.write(f"- {elem}: {count} atoms")

    # Approximate molecular weight
    try:
        from rdkit.Chem import Descriptors
        smiles = formula_to_smiles(formula)
        mol = prepare_molecule(smiles)
        if mol:
            weight = Descriptors.MolWt(mol)
            st.write(f"**Approximate Molecular Weight:** {weight:.2f} g/mol")
    except Exception as e:
        st.warning(f"Could not calculate molecular weight: {e}")
def show_advanced_analysis(formula: str, style: str, colorscheme: str = "Jmol"):
    smiles = formula_to_smiles(formula)
    if smiles:
        mol = prepare_molecule(smiles)
        if mol:
            try:
                mb = Chem.MolToMolBlock(mol)
                viewer = py3Dmol.view(width=800, height=600)
                viewer.addModel(mb, 'mol')
                viewer.setStyle(get_render_style(style, colorscheme))  # Now works
                viewer.setBackgroundColor("white")
                viewer.zoomTo()
                html = viewer._make_html()
                st.components.v1.html(html, height=600, width=800)
            except Exception as e:
                st.error(f"3D visualization error: {e}")
                st.info("Fallback: showing 2D")
                try:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
                except:
                    st.warning("2D rendering failed.")
        else:
            st.warning("Could not prepare molecule.")
    else:
        st.warning("SMILES conversion failed.")
def calculate_properties(formula: str) -> Dict:
    """Calculate molecular properties with robust conformer handling"""
    comp = Composition(formula)
    elements = {str(e): round(comp.get_atomic_fraction(e)*100, 2) for e in comp.elements}
    props = {}
    smiles = formula_to_smiles(formula)
    
    if smiles:
        mol = prepare_molecule(smiles)
        if mol and mol.GetNumConformers() > 0:
            try:
                props.update({
                    'logp': round(Descriptors.MolLogP(mol), 2),
                    'tpsa': round(Descriptors.TPSA(mol), 2),
                    'mol_weight': round(Descriptors.MolWt(mol), 2),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'h_donors': Descriptors.NumHDonors(mol),
                    'h_acceptors': Descriptors.NumHAcceptors(mol),
                    'formal_charge': Chem.GetFormalCharge(mol)
                })
            except Exception as e:
                st.warning(f"Property calc fallback (2D): {e}")
    
    return {
        'mass': round(comp.weight, 4),
        'elements': elements,
        'properties': props,
        'info': get_compound_info(formula)
    }
def show_3d_molecule(smiles: str, style: str = 'Stick', colorscheme: str = 'Jmol', width: int = 800, height: int = 600) -> None:
    """Display 3D molecule with robust conformer check, dynamic styling, and 2D fallback."""
    mol = prepare_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES — cannot prepare molecule.")
        return

    try:
        # Ensure coordinates exist (fallback to 2D if not)
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)

        mb = Chem.MolToMolBlock(mol)
        if not mb or "V2000" not in mb:
            raise ValueError("Invalid MolBlock generated")

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mb, 'mol')
        viewer.setStyle(get_render_style(style, colorscheme))
        viewer.setBackgroundColor("white")  # You can make this dynamic too
        viewer.zoomTo()

        # Hover labels
        viewer.setHoverable({}, True,
            '''
            function(atom,viewer,event,container) {
                if(!atom.label) {
                    atom.label = viewer.addLabel(
                        "Atom " + atom.serial + ": " + atom.elem,
                        {
                            position: atom,
                            backgroundColor: "black",
                            fontColor: "white",
                            fontSize: 14,
                            inFront: true
                        }
                    );
                }
            }''',
            '''
            function(atom,viewer) {
                if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                }
            }'''
        )

        # Render in Streamlit
        st.components.v1.html(viewer._make_html(), height=height)

    except Exception as e:
        st.error(f"⚠️ 3D visualization error: {str(e)}")
        st.info("Tip: Try switching to 2D view or simplifying the molecule structure.")

        try:
            # Fallback to 2D image
            img = Draw.MolToImage(mol, size=(300, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
        except Exception as e2:
            st.error("2D fallback failed too. Try a simpler molecule.")

import hashlib
# ----------------------
# Cached optimizer (keyed by immutable tuples => safe for Streamlit cache)
# ----------------------
@st.cache_resource
def optimize_positions_cached(pos_tuple: tuple, symbols: tuple, fmax: float = 0.05, steps: int = 200):
    """
    pos_tuple: tuple of (x,y,z) tuples  -- hashable
    symbols: tuple of element symbols     -- hashable
    Returns: (optimized_positions (np.array), forces (np.array), energy_ev (float))
    """
    positions = np.array(pos_tuple, dtype=float)
    ase_atoms = Atoms(list(symbols), positions=positions)
    ase_atoms.set_calculator(EMT())

    # Run BFGS, quiet output
    dyn = BFGS(ase_atoms, logfile=None)
    try:
        dyn.run(fmax=fmax, steps=steps)
    except Exception:
        # If it fails to converge, return current state (no crash)
        pass

    new_positions = ase_atoms.get_positions()
    forces = ase_atoms.get_forces()
    try:
        energy_ev = ase_atoms.get_potential_energy()
    except Exception:
        energy_ev = float("nan")

    return new_positions, forces, energy_ev


# ----------------------
# Main view
# ----------------------
def quantum_calculations_view(formula: str) -> None:
    """Run safe geometry optimization with before/after comparison and metrics."""

    # --- Convert formula to SMILES ---
    smiles = formula_to_smiles(formula)
    if not smiles:
        st.warning("SMILES conversion required for quantum analysis")
        return

    # --- Prepare molecule ---
    mol = prepare_molecule(smiles)
    if not mol:
        st.warning("Invalid molecule structure")
        return

    if mol.GetNumConformers() == 0:
        st.warning("No 3D conformer available for calculation")
        return

    # --- Original data ---
    original_coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
    symbols = tuple(atom.GetSymbol() for atom in mol.GetAtoms())
    initial_xyz = Chem.MolToXYZBlock(mol)

    # --- Run button ---
    run_now = st.sidebar.button("▶️ Run Geometry Optimization")
    result_key = "quantum_result_" + hashlib.sha256(initial_xyz.encode()).hexdigest()
    prev = st.session_state.get(result_key, None)

    if run_now or prev is None:
        with st.spinner("Optimizing geometry..."):
            try:
                # --- RDKit optimization ---
                mol_copy = Chem.Mol(mol)
                try:
                    mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy)
                    if mmff_props:
                        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props)
                        ff.Minimize()
                        energy_ev = ff.CalcEnergy()
                        energy_kj = energy_ev * 96.485  # eV → kJ/mol
                    else:
                        AllChem.UFFOptimizeMolecule(mol_copy)
                        energy_ev = None
                        energy_kj = None
                except:
                    AllChem.UFFOptimizeMolecule(mol_copy)
                    energy_ev = None
                    energy_kj = None

                # --- Optimized positions ---
                optimized_coords = np.array(mol_copy.GetConformer().GetPositions(), dtype=float)
                optimized_xyz = Chem.MolToXYZBlock(mol_copy)

                # RMSD
                rmsd = np.sqrt(np.mean(np.sum((original_coords - optimized_coords) ** 2, axis=1)))

                # Forces (approximate)
                force_magnitudes = np.linalg.norm(original_coords - optimized_coords, axis=1)

                # Store results
                st.session_state[result_key] = {
                    "initial_xyz": initial_xyz,
                    "optimized_xyz": optimized_xyz,
                    "new_positions": optimized_coords,
                    "force_magnitudes": force_magnitudes,
                    "energy_ev": energy_ev,
                    "energy_kj": energy_kj,
                    "rmsd": rmsd,
                    "optimized_mol": mol_copy,
                }
                prev = st.session_state[result_key]

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                return

    # --- Display metrics ---
    if prev:
        col1, col2, col3 = st.columns(3)
        with col1:
            ev, kj = prev["energy_ev"], prev["energy_kj"]
            if ev is not None and kj is not None:
                st.metric("Optimized Energy", f"{ev:.4f} eV\n({kj:.2f} kJ/mol)")
            else:
                st.metric("Optimized Energy", "Not available (force-field only)")

        with col2:
            st.metric("Number of Atoms", f"{len(symbols)}")

        with col3:
            st.metric("Geometry RMSD", f"{prev['rmsd']:.4f} Å")

        # --- Forces plot ---
        fig = px.bar(
            x=list(symbols),
            y=prev["force_magnitudes"],
            labels={"x": "Atoms", "y": "Force Magnitude (Å)"},
            title="Atomic Force Distribution (approx.)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 3D viewer ---
        with st.expander("🧪 View Molecular Geometry (Before vs After)", expanded=True):
            view_option = st.radio(
                "Select structure to view:",
                ["Before Optimization", "After Optimization"],
                horizontal=True
            )
            sdf_data = Chem.MolToMolBlock(mol if view_option == "Before Optimization" else prev["optimized_mol"])
            view = py3Dmol.view(width=550, height=420)
            view.addModel(sdf_data, "sdf")
            view.setStyle({"stick": {"radius": 0.15}})
            view.setBackgroundColor("white")
            view.zoomTo()
            st.components.v1.html(view._make_html(), height=420)

        # --- Downloads ---
        st.markdown("### 📥 Download XYZ Files")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️ Before Optimization", prev["initial_xyz"], file_name="before_optimization.xyz")
        with dl2:
            st.download_button("⬇️ After Optimization", prev["optimized_xyz"], file_name="after_optimization.xyz")

        # --- Bond lengths ---
        with st.expander("📏 Bond Lengths (After Optimization)", expanded=False):
            seen = set()
            opt_mol = prev["optimized_mol"]
            coords = prev["new_positions"]
            for bond in opt_mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                pair = tuple(sorted((i, j)))
                if pair in seen:
                    continue
                seen.add(pair)
                ai = opt_mol.GetAtomWithIdx(i).GetSymbol()
                aj = opt_mol.GetAtomWithIdx(j).GetSymbol()
                dist = np.linalg.norm(coords[i] - coords[j])
                st.write(f"{ai}-{aj}: {dist:.3f} Å")

# =============================================================================
# Chem Q&A Bot (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def ask_chemistry_bot(query: str) -> str:
    """
    Recognize 'formula of X' / 'properties of X' and fetch quick data from PubChem.
    Cached to reduce API traffic.
    """
    m_prop = re.search(r"properties of ([\w\s\-]+)", query, re.I)
    m_form = re.search(r"formula of ([\w\s\-]+)", query, re.I)
    name = m_prop.group(1).strip() if m_prop else m_form.group(1).strip() if m_form else None

    if name:
        try:
            comps = pcp.get_compounds(name, "name")
            if not comps:
                return f"❌ I couldn’t find anything for “{name}” in PubChem."
            c = comps[0]
            formula = c.molecular_formula or "Unknown"
            mw = getattr(c, "molecular_weight", None)
            try:
                mw_val = float(mw)
                mw_str = f"{mw_val:.2f} g/mol"
            except (ValueError, TypeError):
                mw_str = "Unknown"

            if m_form:
                return f"The chemical formula for **{name.title()}** is **{formula}**."

            parts = [
                f"**Compound:** {name.title()}",
                f"Formula: {formula}",
                f"Molecular Weight: {mw_str}",
            ]
            if c.iupac_name:
                parts.append(f"IUPAC: {c.iupac_name}")
            if c.synonyms:
                parts.append(f"Synonyms: {', '.join(c.synonyms[:3])}")
            return " • ".join(parts)
        except Exception as e:
            return f"❌ PubChem lookup failed: {e}"

    return (
        "🤖 I couldn't parse a compound name. Try:\n"
        "- 'formula of sulfuric acid'\n"
        "- 'properties of glucose'"
    )

# =============================================================================
# Chemistry Bot Voice/Text Sidebar
# =============================================================================

def chemistry_voice_assistant():
    """Sidebar voice/text gateway to ask_chemistry_bot()."""
    st.markdown("### 🤖 Ask Chemistry Bot")
    st.caption("Try saying: 'Formula of sulfuric acid' or 'Properties of glucose'.")

    # Toggle between voice and text mode
    voice_input_enabled = st.checkbox("🎙️ Voice Mode", value=False, key="use_voice_toggle_sidebar")
    if voice_input_enabled:
        # --- Option A: Live Mic Recording ---
        if st.button("🎤 Record Voice Question", key="record_btn_sidebar"):
            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    with st.spinner("🎧 Listening..."):
                        audio = recognizer.listen(source, timeout=5)

                user_query = recognizer.recognize_google(audio)
                st.success(f"💬 You asked: `{user_query}`")

                answer = ask_chemistry_bot(user_query)
                st.markdown(f"**🧠 Assistant:** {answer}")
                speak_text(answer, voice_type="male")

            except sr.WaitTimeoutError:
                st.warning("⚠️ Listening timed out. Try again.")
            except sr.UnknownValueError:
                st.error("❌ Sorry, I couldn't understand. Please try speaking clearly.")
            except Exception as e:
                st.error(f"🎙️ Microphone error: {e}")

        # --- Option B: Upload Audio File ---
        audio_file = st.file_uploader("📂 Or upload a recorded question", type=["wav", "mp3", "m4a"])
        if audio_file:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            try:
                user_query = recognizer.recognize_google(audio)
                st.success(f"💬 You asked: `{user_query}`")
                answer = ask_chemistry_bot(user_query)
                st.markdown(f"**🧠 Assistant:** {answer}")
                speak_text(answer, voice_type="male")
            except Exception as e:
                st.error(f"Audio recognition failed: {e}")

    else:
        # --- Fallback: Text Input ---
        user_query = st.text_input("⌨️ Type your chemistry question:", key="text_input_sidebar")
        if user_query:
            st.markdown(f"💬 You asked: `{user_query}`")
            answer = ask_chemistry_bot(user_query)
            st.markdown(f"**🧠 Assistant:** {answer}")
            speak_text(answer, voice_type="male")


from typing import Optional, Tuple
from functools import lru_cache

# Make sure CHEMICAL_DATABASE, get_pubchem_compounds, and VOICE_ENABLED / ASE_ENABLED are already defined

@st.cache_data
def cached_pubchem(formula: str):
    """Cache PubChem fetches to speed up repeated queries."""
    return get_pubchem_compounds(formula)

def sidebar_controls() -> Tuple[str, str, str, Optional[str], Optional[str]]:
    """Create optimized sidebar controls for Molecular Explorer Pro."""
    with st.sidebar:
        # -----------------------------
        # About section
        # -----------------------------
        st.markdown("""
        <div style="background:#f0f2f6; padding:1rem; border-radius:8px;">
            <h3>🚀 About</h3>
            <p><strong>Molecular Explorer Pro</strong></p>
            <p>Developed by Aqsa Ijaz</p>
            <p>Interactive chemistry visualization & analysis</p>
            <p><em>Version 5.6</em></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # -----------------------------
        # Main controls
        # -----------------------------
        st.title("⚗️ Controls")

        view_mode = st.radio(
            "Analysis Mode:",
            [
                "🔍 Quick Analysis",
                "📊 Detailed Report",
                "🔄 3D Explorer",
                "🔬 Quantum Calc"
            ],
            index=0
        )

        st.markdown("---")

        # -----------------------------
        # Optional features
        # -----------------------------
        st.markdown("### 🧠 Optional Features")

        # Voice assistant toggle
        voice_toggle = st.checkbox("🎙️ Enable voice assistant (local only)", value=False)
        st.session_state["voice_allowed"] = voice_toggle and VOICE_ENABLED

        # ASE toggle
        ase_toggle = st.checkbox("⚛️ Enable geometry optimization (ASE)", value=False)
        st.session_state["ase_allowed"] = ase_toggle and ASE_ENABLED

        st.markdown("---")

        # -----------------------------
        # Compound selection
        # -----------------------------
        st.markdown("### 🧪 Select Compound")
        query = st.text_input("Search compound (local search, PubChem optional):", "")

        # Local database matches
        local_matches = [
            name for name in CHEMICAL_DATABASE.keys()
            if query.lower() in name.lower() or query.lower() in CHEMICAL_DATABASE[name].lower()
        ] or list(CHEMICAL_DATABASE.keys())  # fallback to all

        compound = st.selectbox("Choose from local matches:", local_matches, index=0)
        formula = CHEMICAL_DATABASE[compound]

        # Optional PubChem fetch (on-demand)
        if query.strip() and st.button("Fetch PubChem suggestions"):
            pubchem_results = cached_pubchem(query)
            if pubchem_results:
                st.markdown("**PubChem suggestions:**")
                for c in pubchem_results[:5]:  # top 5 suggestions
                    st.write(f"{c.iupac_name or 'Unknown'} ({c.molecular_formula})")

        # -----------------------------
        # 3D Explorer options
        # -----------------------------
        render_style = None
        bg_color = None
        if view_mode == "🔄 3D Explorer":
            render_style = st.selectbox(
                "🧬 Render Style",
                ["Stick", "Sphere", "Cartoon", "Surface"]
            )
            bg_color = st.color_picker(
                "🎨 Background Color",
                "#FFFFFF"
            )

        return view_mode, compound, formula, render_style, bg_color
    

# Initialize Text-to-Speech engine globally
engine = pyttsx3.init()

def speak(text):
    try:
        if not engine._inLoop:
            engine.say(text)
            engine.runAndWait()
    except RuntimeError as e:
        print(f"[Voice Error]: {e}")

# --- Bot Query Function ---
@st.cache_data(show_spinner=False)
def ask_chemistry_bot(query: str) -> str:
    m_prop = re.search(r'properties of ([\w\s]+)', query, re.I)
    m_form = re.search(r'formula of ([\w\s]+)', query, re.I)
    name = None
    if m_prop:
        name = m_prop.group(1).strip()
    elif m_form:
        name = m_form.group(1).strip()

    if name:
        try:
            comps = pcp.get_compounds(name, 'name')
            if not comps:
                return f"❌ I couldn’t find anything for “{name}” in PubChem."
            c = comps[0]
            formula = c.molecular_formula or 'Unknown'
            mw = getattr(c, 'molecular_weight', None)
            try:
                mw_val = float(mw)
                mw_str = f"{mw_val:.2f} g/mol"
            except (ValueError, TypeError):
                mw_str = "Unknown"

            if m_form:
                return f"The chemical formula for **{name.title()}** is **{formula}**."

            props = [
                f"**Compound:** {name.title()}",
                f"Formula: {formula}",
                f"Molecular Weight: {mw_str}"
            ]
            if c.iupac_name:
                props.append(f"IUPAC Name: {c.iupac_name}")
            if c.synonyms:
                props.append(f"Synonyms: {', '.join(c.synonyms[:3])}")
            return " • ".join(props)
        except Exception as e:
            return f"❌ PubChem lookup failed: {e}"

    return "🤖 I couldn't find a compound name in your question. Try asking like:\n- 'formula of sulfuric acid'\n- 'properties of glucose'"

#    """Main application function"""
def main():
    st.markdown("""
        <style>
            .header {
                background: linear-gradient(135deg, #1e88e5, #0d47a1);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .creator-credit {
                color: #00e5ff;
                font-weight: bold;
                font-size: 1.2em;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header">
            <h1 style="text-align:center;">🧪 Molecular Explorer Pro</h1>
            <p style="text-align:center;">
                Interactive Chemistry Analysis Tool — by <span class="creator-credit">Aqsa Ijaz ✨</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load sidebar controls
    view_mode, compound, formula, render_style, bg_color = sidebar_controls()

    # Render different views based on selection
    if view_mode == "🔍 Quick Analysis":
        quick_analysis_view(formula)
    elif view_mode == "📊 Detailed Report":
        detailed_report_view(formula)
    elif view_mode == "🔄 3D Explorer":
        three_d_explorer_view(formula, render_style, bg_color)
    elif view_mode == "🔬 Quantum Calc":
        if is_quantum_supported(formula):
            quantum_calculations_view(formula)
        else:
            st.warning("❌ Quantum calculations not supported for this molecule.")
    from sascorer import readFragmentScores
    readFragmentScores()
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Molecular Explorer Pro v5.6 • Created by Aqsa Ijaz • © 2024
        </div>
    """, unsafe_allow_html=True)

def quick_analysis_view(formula: str) -> None:
    """Display quick analysis view with enhanced visualization"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🧮 Basic Properties")
        results = calculate_properties(formula)
        if results:
            st.metric("Molecular Mass", f"{results['mass']} g/mol")
            if results['elements']:
                fig, ax = plt.subplots(figsize=(6, 6))
                wedges, texts = ax.pie(
                    results['elements'].values(),
                    labels=results['elements'].keys(),
                    startangle=90,
                    textprops={'fontsize': 10}
                )
                ax.axis('equal')
                st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🧪 Compound Info")
        info = get_compound_info(formula)
        st.write(f"**Name**: {info['name'].title()}")
        st.write(f"**Formula**: {info['formula']}")
        smiles = formula_to_smiles(formula)
        # Skip any 3D-related messages completely
st.markdown("</div>", unsafe_allow_html=True)
def detailed_report_view(formula: str) -> None:
    """Display detailed report view with enhanced visualization"""
    tab1, tab2, tab3 = st.tabs(["📈 Properties", "🧪 Composition", "🔬 Advanced"])

    # Tab 1: Detailed Properties
    with tab1:
        st.subheader("📊 Detailed Molecular Properties")
        results = calculate_properties(formula)
        if not results:
            st.warning("No property data available.")
            return

        col1, col2 = st.columns(2)
        with col1:
            fc = results['properties'].get('formal_charge')
            if fc is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>⚡ Formal Charge:</b> {fc}<br>
                        <small style='color:gray;'>Overall net charge</small>
                    </div>
                """, unsafe_allow_html=True)

            logp = results['properties'].get('logp')
            if logp is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>💧 LogP (Hydrophobicity):</b> {logp}<br>
                        <small style='color:gray;'>Higher = fat‑soluble</small>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            tpsa = results['properties'].get('tpsa')
            if tpsa is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>🧲 TPSA (Polar Surface Area):</b> {tpsa} Å²<br>
                        <small style='color:gray;'>Polar interaction potential</small>
                    </div>
                """, unsafe_allow_html=True)

            rotb = results['properties'].get('rotatable_bonds')
            if rotb is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>🔗 Bond Flexibility:</b> {rotb} rotatable bonds<br>
                        <small style='color:gray;'>Higher = more flexible</small>
                    </div>
                """, unsafe_allow_html=True)

    # Tab 2: Elemental Composition (Correctly Placed)
    with tab2:
        st.subheader("🧪 Elemental Composition Analysis")
        results = calculate_properties(formula)
        elems = results.get('elements', {})
        if not elems:
            st.warning("No composition data available.")
            return

        # 🌈 Sparkly Plotly Donut Chart
        fig = go.Figure(data=[go.Pie(
            labels=list(elems.keys()),
            values=list(elems.values()),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=13,
            marker=dict(
                colors=px.colors.qualitative.Pastel,
                line=dict(color='#FFFFFF', width=2)
            )
        )])

        fig.update_layout(
            title={
                'text': "Elemental Composition",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=30, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # 📋 Pretty Composition Table
        df = pd.DataFrame([
            {
                "Symbol": str(el),
                "Count": Composition(formula)[el],
                "Atomic Mass": f"{el.atomic_mass:.4f}",
                "Electronegativity": f"{el.X:.2f}"
            }
            for el in Composition(formula).elements
        ])

        st.markdown("### 📋 Element Summary Table")
        st.dataframe(df.style.background_gradient(cmap="YlGnBu", subset=["Electronegativity"]))

    # Tab 3: Advanced Metrics
    with tab3:
        st.subheader("🔬 Advanced Molecular Analysis")
        smiles = formula_to_smiles(formula)
        if not smiles:
            st.warning("SMILES conversion failed for advanced metrics.")
            return

        mol = prepare_molecule(smiles)
        if mol is None:
            st.warning("Molecule preparation failed for advanced metrics.")
            return

        # Lipinski Rule of 5
        lip = evaluate_lipinski(mol)
        st.markdown("#### ✅ Lipinski Rule of 5")
        for rule, ok in lip.items():
            if rule != "passed":
                st.markdown(f"- {rule.replace('_',' ').title()}: {'✔️' if ok else '❌'}")
        st.success("Passes all Lipinski criteria!" if lip["passed"] else "❌ Does not satisfy all Lipinski criteria")

        # Bioavailability Score
        bio = estimate_bioavailability(mol)
        st.markdown(f"#### 🧬 Bioavailability Score: **{bio:.2f}**")
        st.caption("Higher → more likely to be orally active")

        # Synthetic Accessibility
        try:
            sas = calculate_sascore(mol)
            st.markdown(f"#### 🛠️ Synthetic Accessibility: **{sas:.2f}**")
            st.caption("1 = easy to synthesize, 10 = very difficult")
        except Exception as e:
            st.warning(f"Could not compute SAScore: {e}")

def three_d_explorer_view(formula: str, render_style: str = "Stick", bg_color: str = "#ffffff") -> None:
    """Enhanced 3D molecule explorer with safe styling, hover, auto-rotate, and optional snapshot."""
    smiles = formula_to_smiles(formula)
    if not smiles:
        st.error("❌ Could not get SMILES for 3D view.")
        return

    mol = prepare_molecule(smiles)
    if mol is None:
        st.error("❌ Could not prepare molecule for 3D view.")
        return

    # --- UI Chrome ---
    st.markdown("""
    <style>
    .three-d-container {
        background: linear-gradient(145deg, #f0f2f5, #d9e4f5);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .three-d-title {
        font-size: 1.8rem;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="three-d-container">', unsafe_allow_html=True)
    st.markdown('<div class="three-d-title">🔄 3D Molecule Explorer</div>', unsafe_allow_html=True)

    # --- Controls ---
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        style = st.selectbox("🧬 Style:", ["Stick", "Sphere", "Cartoon", "Surface"],
                             index=["Stick", "Sphere", "Cartoon", "Surface"].index(render_style)
                             if render_style in ["Stick", "Sphere", "Cartoon", "Surface"] else 0)
    with col2:
        colorscheme = st.selectbox("🎨 Colorscheme:", ["default", "Jmol", "chain", "element"], index=1)
    with col3:
        bg_color = st.color_picker("🖼️ Background Color:", value=bg_color)

    col4, col5 = st.columns([1, 1])
    with col4:
        rotate = st.toggle("🔄 Auto-Rotate", value=True)
    with col5:
        enable_snapshot = st.toggle("📸 Enable Snapshot Button", value=False)

    # --- Build 3D View ---
    try:
        mb = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=700, height=500)
        viewer.addModel(mb, 'mol')

        # Main style
        main_style = get_render_style(style, colorscheme)
        viewer.setStyle(main_style)

        # If cartoon selected but nothing would render, layer stick fallback
        if style == "Cartoon":
            viewer.addStyle({}, {"stick": {"radius": 0.15, "colorscheme": colorscheme}})

        viewer.setBackgroundColor(bg_color)
        viewer.zoomTo()

        if rotate:
            viewer.spin(True)

        if enable_snapshot:
            viewer.addButton("📸 Snapshot", """
                function(){
                  viewer.render();
                  viewer.downloadImage({format:'png', backgroundColor:'white'});
                }
            """)

        # Hover labels
        viewer.setHoverable({}, True,
            '''
            function(atom,viewer,event,container) {
                if(!atom.label) {
                    atom.label = viewer.addLabel(
                        "Atom " + atom.serial + ": " + atom.elem,
                        {
                            position: atom,
                            backgroundColor: "black",
                            fontColor: "white",
                            fontSize: 14,
                            inFront: true
                        }
                    );
                }
            }''',
            '''
            function(atom,viewer) {
                if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                }
            }'''
        )

        st.components.v1.html(viewer._make_html(), height=500)

    except Exception as e:
        st.error(f"Rendering failed: {e}")
        st.info("Trying 2D view as fallback...")
        try:
            img = Draw.MolToImage(mol, size=(300, 300))
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
        except Exception:
            st.error("2D fallback failed too.")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()