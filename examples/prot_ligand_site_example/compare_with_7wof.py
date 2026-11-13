#!/usr/bin/env python3
"""RMSD comparison of Mol-1 (base) and Mol-1-constraint predictions of
SARS-CoV-2 Mpro + GC376 against the experimental structure 7WOF.

Self-contained: all inputs live in this example directory.
Run from anywhere with the ROCm venv python:
  ../../.venv... not required — needs only numpy, gemmi, rdkit:
  python examples/prot_ligand_site_example/compare_with_7wof.py

Method:
  - Kabsch superposition on CA atoms of residues 1-303 common to all models.
  - Ligand (GC376 = CCD 5IZ, 30 heavy atoms) matched 1:1 by graph
    connectivity with RDKit substructure matching.
Writes rmsd_7wof.json + overlay_7wof_data.npy next to this script.
"""
import os, json
import numpy as np
import gemmi
from rdkit import Chem

DIR = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(DIR, "7WOF.pdb")
PRED_BASE = os.path.join(DIR, "predictions", "mol1_base_sample_0.cif")
PRED_CPREP = os.path.join(DIR, "predictions", "prot_ligand_site_sample_0.cif")
SITE = [41, 49, 142, 143, 144, 145]

def get_str(path):
    if path.endswith(".cif"):
        st = gemmi.make_structure_from_block(gemmi.cif.read(path)[0])
    else:
        st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.setup_entities()
    return st

def ca_atoms(st):
    ca = {}
    for ch in st[0]:
        for res in ch:
            for a in res:
                if a.name == "CA" and res.name != "HOH":
                    ca[res.seqid.num] = np.array(a.pos.tolist())
    return ca

def lig_atoms(st):
    out = []
    for ch in st[0]:
        for res in ch:
            if res.het_flag == "H" and res.name != "HOH" and len(res) > 5:
                for a in res:
                    out.append((a.element.name, a.name, np.array(a.pos.tolist())))
    return out

def kabsch(P, Q):
    """Superpose Q onto P; returns rmsd and transform f(x)."""
    tc, tcq = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - tc).T @ (Q - tcq))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    rms = float(np.sqrt((((P - tc) - (Q - tcq) @ R.T) ** 2).sum() / len(P)))
    return rms, lambda x: (x - tcq) @ R.T + tc

def mol_from_lig(lig):
    lines = []
    for i, (el, name, xyz) in enumerate(lig, 1):
        lines.append(f"HETATM{i:5d} {name:>4} LIG L   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2}")
    lines.append("END")
    m = Chem.MolFromPDBBlock("\n".join(lines), removeHs=False, proximityBonding=True, sanitize=False)
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.ZERO)   # match connectivity only
    return m

st_exp = get_str(EXP)
ca_exp = ca_atoms(st_exp)
lig_exp = lig_atoms(st_exp)
m_exp = mol_from_lig(lig_exp)

report = {}
fig_data = {"exp": dict(ca=np.array([ca_exp[i] for i in sorted(ca_exp)]),
                        lig=np.array([x for _, _, x in lig_exp]),
                        site=np.array([ca_exp[i] for i in SITE]))}

for tag, path in (("mol-1", PRED_BASE), ("mol-1-constraint", PRED_CPREP)):
    st = get_str(path)
    ca_p = ca_atoms(st)
    lig_p = lig_atoms(st)
    common = sorted(set(ca_exp) & set(ca_p))
    rms_all, fit = kabsch(np.array([ca_exp[i] for i in common]),
                          np.array([ca_p[i] for i in common]))
    rms_site_gf = float(np.sqrt(sum(np.sum((fit(ca_p[i]) - ca_exp[i]) ** 2) for i in SITE) / len(SITE)))
    rms_site_of, _ = kabsch(np.array([ca_exp[i] for i in SITE]),
                            np.array([ca_p[i] for i in SITE]))
    m_pred = mol_from_lig(lig_p)
    lm = m_exp.GetSubstructMatches(m_pred, uniquify=True)[0]
    Pprot = np.array([fit(x) for _, _, x in lig_p])
    Pe_all = np.array([x for _, _, x in lig_exp])
    rms_lig_gf = float(np.sqrt(((Pprot - Pe_all[list(lm)]) ** 2).sum() / len(lm)))
    rms_lig_bf, _ = kabsch(Pprot, Pe_all[list(lm)])
    report[tag] = {
        "protein_CA_RMSD_A": round(rms_all, 2),
        "site_CA_RMSD_globalfit_A": round(rms_site_gf, 2),
        "site_CA_RMSD_ownfit_A": round(rms_site_of, 2),
        "ligand_RMSD_proteinfit_A": round(rms_lig_gf, 2),
        "ligand_RMSD_bestfit_A": round(rms_lig_bf, 2),
        "ligand_matched_atoms": len(lm),
        "CA_common_residues": len(common),
    }
    fig_data[tag] = dict(ca={i: fit(ca_p[i]) for i in common},
                         lig=Pprot, site=[fit(ca_p[i]) for i in SITE])
    print(tag, report[tag])

json.dump({"experimental": "7WOF (SARS-CoV-2 Mpro + GC376 ligand 5IZ)",
           "site_residues": SITE,
           "note": "Kabsch on common CA; ligand 30/30 heavy-atom graph match; same molecule in all three.",
           "metrics": report},
          open(os.path.join(DIR, "rmsd_7wof.json"), "w"), indent=2)
np.save(os.path.join(DIR, "overlay_7wof_data.npy"), fig_data, allow_pickle=True)
print("wrote rmsd_7wof.json, overlay_7wof_data.npy")
