"""piCalculax is a library for isoelectric point prediction for chemically modified peptides.

The original library has been adapted to be incorporated into Oloren ChemEngine

For more information visit the github repository: https://github.com/EBjerrum/pICalculax
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from numpy import unique, argmin
from numpy import abs as np_abs

# RuleTables
from rdkit import Chem

# Format of ruletable tuple (Name, SMARTS, pKa (evt. as list), type (evt.as list) )
# Type is 0 = acidic (charge below pKa)
# Type 1 is basic (charge below pKa)

# (Grimsley 2009) with extensions
# Grimsley, G. R.; Scholtz, J. M. & Pace, C. N. (2009), 'A summary of the measured pK values of the ionizable groups in folded proteins.', Protein science : a publication of the Protein Society 18, 247--251.

# Default pKa tables defined at the end of the file

pkatable_grimsley = [
    ('Tetramethylrhodamine #3',
     Chem.MolFromSmiles('CN(C)C1=C-C=C2-C(C4=C(C(=O)O)C=CC=C4)=C3-C=C-C(=[N+](C)C)C=C3OC2=C1'), [-10, 1.19, 3.59, 10],
     [1, 1, 0, 1]),
    ('6-hydroxy-3H-xanthen-3-one', Chem.MolFromSmarts('c2c3ccc(O)cc3oc3cc(=O)ccc2-3'), [2.94, 6.56], [1, 0]),
    ('Anilin', Chem.MolFromSmarts('[$([NH2]c1ccccc1)]'), 4.87, 1),
    ('N-acetyl Hydrazine', Chem.MolFromSmarts('[$([NH2]NC(=O))]'), 2.81, 1),  # Reaxys
    ('Hydrazine', Chem.MolFromSmarts('[$([NH2]N)]'), 8.1, 1),  # Wikiped
    ('N-acetyl Piperazine', Chem.MolFromSmarts('[N+&!H0,NX3&H0;$(N1(C)CCN(C(=O)C)CC1)]'), 7.1, 1),  # Reaxys
    ('Piperazine', Chem.MolFromSmarts('C1CNCCN1'), 9.73, 1),  # Wikiped
    ('Pyridine', Chem.MolFromSmarts('n1ccccc1'), 5.25, 1),  # Wikiped
    ('Sulfonate', Chem.MolFromSmarts('[$([OH]S(=O)=O)]'), -1.5, 0),  # Guess
    ('Phosphonate', Chem.MolFromSmarts('[OH]P([OH])(=O)[#6]'), [1.5, 7.0], [0, -1]),  # Guess
    ('7H-Pyrimido[4,5-b][1,4]oxazine-2,4-diamine', Chem.MolFromSmarts('C1C=NC2=CN=C(N=C2O1)N'), 3.6, 1),  # PKA Guessed
    ('Tyrosine', Chem.MolFromSmarts('[$([O-,OH]c1ccccc1)]'), 10.3, 0),
    ('Histidine', Chem.MolFromSmarts(
        '[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1'),
     6.6, 1),
    ('Histidine1', Chem.MolFromSmarts('[$(cC)]1cnc[nH]1'), 6.6, 1),
    ('Histidine2', Chem.MolFromSmarts('[$(cC)]1c[nH]cn1'), 6.6, 1),
    ('Glutamic Acid', Chem.MolFromSmarts('[$([CX3][CD2][CD2]C(N)C(=O))](=O)[OX1H0-,OX2H1]'), 4.2, 0),
    ('Cysteine', Chem.MolFromSmarts('[$([S-,SH][CD2]C(N)C(=O))]'), 6.8, 0),
    ('Condensed Cysteine', Chem.MolFromSmarts('[$([SH][#175])]'), 6.8, 0),  # Sidechain treated as normal atom
    ('Condensed Cysteine', Chem.MolFromSmarts('[$([SH][#178])]'), 6.8, 0),  # Sidechain treated as normal atom
    # ('Aspartic Acid', Chem.MolFromSmarts('[$([CX3][CD2])](=O)[OX1H0-,OX2H1]'),3.90,0),
    ('Aspartic Acid v2', Chem.MolFromSmarts('[$([CX3][CD2]C(N)C(=O))](=O)[OX1H0-,OX2H1]'), 3.5, 0),
    ('Arginine', Chem.MolFromSmarts('[$(N[CD2][CD2])]C(=N)N'), 12.0, 1),
    ('Lysine', Chem.MolFromSmarts('[$([ND1][CD2][CD2][CD2][CD2]C(N)C(=O))]'), 10.5, 1),
    ('End Carboxylate (avg)', Chem.MolFromSmarts('[$([CX3]CN)](=O)[OX1H0-,OX2H1]'), 3.3, 0),
    # ('End Amine (avg)',Chem.MolFromSmarts('[$([NX3;H2,H1;!$(NC=O)]CC=O)]'),9.15,1),
    ('End Amine (avg) PseudoAtom safe', Chem.MolFromSmarts(
        '[$([NX3;H2,H1;!$(NC=O);!$(N[#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193])]CC=O)]'),
     7.7, 1),
    ('Acrydine', Chem.MolFromSmarts('c1c2ccccc2nc2ccccc21'), 6.15, 1),
    ('Generic Carboxylic Acid', Chem.MolFromSmarts('[CX3](=O)[OX1H0-,OX2H1]'), 4.75, 0),
    (
    'Generic tert. Amine, not amide', Chem.MolFromSmarts('[N+&!H0&NX3,NX3;$(N(C)(C)C);!$(NC=[!#6]);!$(NC#[!#6])]'), 9.8,
    1),
    ('Generic Amine, not amide', Chem.MolFromSmarts(
        '[N+,NX3;!H0;!$(NC=[!#6]);!$(NC#[!#6])!$(N[#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193])]'),
     10., 1),
]

condensed_grimsley = [
    ('Condensed Tyrosine', Chem.MolFromSmarts('[#189]'), 10.3, 0),
    ('Condensed Histidine', Chem.MolFromSmarts('[#179]'), 6.6, 1),
    ('Condensed Glutamic Acid', Chem.MolFromSmarts('[#177]'), 4.2, 0),
    ('Condensed Aspartic Acid', Chem.MolFromSmarts('[#174]'), 3.5, 0),
    ('Condensed Arginine', Chem.MolFromSmarts('[#172]'), 12.0, 1),
    ('Condensed Lysine', Chem.MolFromSmarts('[#182]'), 10.5, 1),
    # ('Condensed AA end amino subst.', Chem.MolFromSmarts('[#0D1]'),9.15,1),
    # Can it somehow match up the wrong end? with a protection group??
    ('Condensed AA end amino', Chem.MolFromSmarts(
        '[#171,#172,#173,#174,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193;D1]'),
     7.7, 1),
    # ('Condensed AA end Carboxylic Subst.', Chem.MolFromSmarts('[$([#0][OH])]'),2.15,0), #All acidic/basic should be replaced with glycine so they can be removed??
    ('Condensed cys end amino', Chem.MolFromSmarts('[#175;D2]'), 7.7, 1),
    # ('Condensed AA end Carboxylic Subst.', Chem.MolFromSmarts('[$([#0][OH])]'),2.15,0), #All acidic/basic should be replaced with glycine so they can be removed??
    ('Condensed AA end Carboxylic', Chem.MolFromSmarts(
        '[$([#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193][OH])]'),
     3.3, 0)
]

# IPC_peptide as refered in KOZLOWSKI, Lukasz P. "IPC - Isoelectric Point Calculator.". Biol. Direct. 2016, vol 11, p. 55.
pkatable_ipc = [
    ('Tetramethylrhodamine #3',
     Chem.MolFromSmiles('CN(C)C1=C-C=C2-C(C4=C(C(=O)O)C=CC=C4)=C3-C=C-C(=[N+](C)C)C=C3OC2=C1'), [-10, 1.19, 3.59, 10],
     [1, 1, 0, 1]),
    # Guess DOI: 10.1039/C0OB01045F
    ('6-hydroxy-3H-xanthen-3-one', Chem.MolFromSmarts('c2c3ccc(O)cc3oc3cc(=O)ccc2-3'), [2.94, 6.56], [1, 0]),
    ('Anilin', Chem.MolFromSmarts('[$([NH2]c1ccccc1)]'), 4.87, 1),
    ('N-acetyl Hydrazine', Chem.MolFromSmarts('[$([NH2]NC(=O))]'), 2.81, 1),  # Reaxys
    ('Hydrazine', Chem.MolFromSmarts('[$([NH2]N)]'), 8.1, 1),  # Wikiped
    ('N-acetyl Piperazine', Chem.MolFromSmarts('[N+&!H0,NX3&H0;$(N1(C)CCN(C(=O)C)CC1)]'), 7.1, 1),  # Reaxys
    ('Piperazine', Chem.MolFromSmarts('C1CNCCN1'), 9.73, 1),  # Wikiped
    ('Pyridine', Chem.MolFromSmarts('n1ccccc1'), 5.25, 1),  # Wikiped
    ('Sulfonate', Chem.MolFromSmarts('[$([OH]S(=O)=O)]'), -1.5, 0),  # Guess
    ('Phosphonate', Chem.MolFromSmarts('[OH]P([OH])(=O)[#6]'), [1.5, 7.0], [0, -1]),  # Guess
    ('7H-Pyrimido[4,5-b][1,4]oxazine-2,4-diamine', Chem.MolFromSmarts('C1C=NC2=CN=C(N=C2O1)N'), 3.6, 1),  # PKA Guessed
    ('Tyrosine', Chem.MolFromSmarts('[$([O-,OH]c1ccccc1)]'), 10.071, 0),
    ('Histidine', Chem.MolFromSmarts(
        '[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1'),
     6.018, 1),
    ('Histidine1', Chem.MolFromSmarts('[$(cC)]1cnc[nH]1'), 6.018, 1),
    ('Histidine2', Chem.MolFromSmarts('[$(cC)]1c[nH]cn1'), 6.018, 1),
    ('Glutamic Acid', Chem.MolFromSmarts('[$([CX3][CD2][CD2]C(N)C(=O))](=O)[OX1H0-,OX2H1]'), 4.317, 0),
    ('Cysteine', Chem.MolFromSmarts('[$([S-,SH][CD2]C(N)C(=O))]'), 8.297, 0),
    ('Condensed Cysteine', Chem.MolFromSmarts('[$([SH][#175])]'), 8.297, 0),  # Sidechain treated as normal atom
    ('Condensed Cysteine', Chem.MolFromSmarts('[$([SH][#178])]'), 8.297, 0),  # Sidechain treated as normal atom
    # ('Aspartic Acid', Chem.MolFromSmarts('[$([CX3][CD2])](=O)[OX1H0-,OX2H1]'),3.90,0),
    ('Aspartic Acid v2', Chem.MolFromSmarts('[$([CX3][CD2]C(N)C(=O))](=O)[OX1H0-,OX2H1]'), 3.887, 0),
    ('Arginine', Chem.MolFromSmarts('[$(N[CD2][CD2])]C(=N)N'), 12.503, 1),
    ('Lysine', Chem.MolFromSmarts('[$([ND1][CD2][CD2][CD2][CD2]C(N)C(=O))]'), 10.517, 1),
    ('End Carboxylate (avg)', Chem.MolFromSmarts('[$([CX3]CN)](=O)[OX1H0-,OX2H1]'), 2.383, 0),
    # ('End Amine (avg)',Chem.MolFromSmarts('[$([NX3;H2,H1;!$(NC=O)]CC=O)]'),9.15,1),
    ('End Amine (avg) PseudoAtom safe', Chem.MolFromSmarts(
        '[$([NX3;H2,H1;!$(NC=O);!$(N[#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193])]CC=O)]'),
     9.564, 1),
    ('Acrydine', Chem.MolFromSmarts('c1c2ccccc2nc2ccccc21'), 6.15, 1),
    ('Generic Carboxylic Acid', Chem.MolFromSmarts('[CX3](=O)[OX1H0-,OX2H1]'), 4.75, 0),
    (
    'Generic tert. Amine, not amide', Chem.MolFromSmarts('[N+&!H0&NX3,NX3;$(N(C)(C)C);!$(NC=[!#6]);!$(NC#[!#6])]'), 9.8,
    1),
    ('Generic Amine, not amide', Chem.MolFromSmarts(
        '[N+,NX3;!H0;!$(NC=[!#6]);!$(NC#[!#6])!$(N[#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193])]'),
     10., 1),
]

condensed_ipc = [
    ('Condensed Tyrosine', Chem.MolFromSmarts('[#189]'), 10.071, 0),
    ('Condensed Histidine', Chem.MolFromSmarts('[#179]'), 6.018, 1),
    ('Condensed Glutamic Acid', Chem.MolFromSmarts('[#177]'), 4.317, 0),
    ('Condensed Aspartic Acid', Chem.MolFromSmarts('[#174]'), 3.887, 0),
    ('Condensed Arginine', Chem.MolFromSmarts('[#172]'), 12.503, 1),
    ('Condensed Lysine', Chem.MolFromSmarts('[#182]'), 10.517, 1),
    # ('Condensed AA end amino subst.', Chem.MolFromSmarts('[#0D1]'),9.15,1),
    # Can it somehow match up the wrong end? with a protection group??
    ('Condensed AA end amino', Chem.MolFromSmarts(
        '[#171,#172,#173,#174,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193;D1]'),
     9.564, 1),
    # ('Condensed AA end Carboxylic Subst.', Chem.MolFromSmarts('[$([#0][OH])]'),2.15,0), #All acidic/basic should be replaced with glycine so they can be removed??
    ('Condensed cys end amino', Chem.MolFromSmarts('[#175;D2]'), 9.564, 1),
    # ('Condensed AA end Carboxylic Subst.', Chem.MolFromSmarts('[$([#0][OH])]'),2.15,0), #All acidic/basic should be replaced with glycine so they can be removed??
    ('Condensed AA end Carboxylic', Chem.MolFromSmarts(
        '[$([#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193][OH])]'),
     2.383, 0)
]

# EMBOSS iep. 2017, http://emboss.sourceforge.net/apps/release/6.6/emboss/apps/iep.html.
pkatable_EMBOSS = [
    ('Tetramethylrhodamine #3',
     Chem.MolFromSmiles('CN(C)C1=C-C=C2-C(C4=C(C(=O)O)C=CC=C4)=C3-C=C-C(=[N+](C)C)C=C3OC2=C1'), [-10, 1.19, 3.59, 10],
     [1, 1, 0, 1]),
    # Guess DOI: 10.1039/C0OB01045F
    ('6-hydroxy-3H-xanthen-3-one', Chem.MolFromSmarts('c2c3ccc(O)cc3oc3cc(=O)ccc2-3'), [2.94, 6.56], [1, 0]),
    ('Anilin', Chem.MolFromSmarts('[$([NH2]c1ccccc1)]'), 4.87, 1),
    ('N-acetyl Hydrazine', Chem.MolFromSmarts('[$([NH2]NC(=O))]'), 2.81, 1),  # Reaxys
    ('Hydrazine', Chem.MolFromSmarts('[$([NH2]N)]'), 8.1, 1),  # Wikiped
    ('N-acetyl Piperazine', Chem.MolFromSmarts('[N+&!H0,NX3&H0;$(N1(C)CCN(C(=O)C)CC1)]'), 7.1, 1),  # Reaxys
    ('Piperazine', Chem.MolFromSmarts('C1CNCCN1'), 9.73, 1),  # Wikiped
    ('Pyridine', Chem.MolFromSmarts('n1ccccc1'), 5.25, 1),  # Wikiped
    ('Sulfonate', Chem.MolFromSmarts('[$([OH]S(=O)=O)]'), -1.5, 0),  # Guess
    ('Phosphonate', Chem.MolFromSmarts('[OH]P([OH])(=O)[#6]'), [1.5, 7.0], [0, -1]),  # Guess
    ('7H-Pyrimido[4,5-b][1,4]oxazine-2,4-diamine', Chem.MolFromSmarts('C1C=NC2=CN=C(N=C2O1)N'), 3.6, 1),  # PKA Guessed
    ('Tyrosine', Chem.MolFromSmarts('[$([O-,OH]c1ccccc1)]'), 10.1, 0),
    ('Histidine', Chem.MolFromSmarts(
        '[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1'),
     6.5, 1),
    ('Histidine1', Chem.MolFromSmarts('[$(cC)]1cnc[nH]1'), 6.5, 1),
    ('Histidine2', Chem.MolFromSmarts('[$(cC)]1c[nH]cn1'), 6.5, 1),
    ('Glutamic Acid', Chem.MolFromSmarts('[$([CX3][CD2][CD2]C(N)C(=O))](=O)[OX1H0-,OX2H1]'), 4.1, 0),
    ('Cysteine', Chem.MolFromSmarts('[$([S-,SH][CD2]C(N)C(=O))]'), 8.5, 0),
    ('Condensed Cysteine', Chem.MolFromSmarts('[$([SH][#175])]'), 8.5, 0),  # Sidechain treated as normal atom
    ('Condensed Cysteine', Chem.MolFromSmarts('[$([SH][#178])]'), 8.5, 0),  # Sidechain treated as normal atom
    # ('Aspartic Acid', Chem.MolFromSmarts('[$([CX3][CD2])](=O)[OX1H0-,OX2H1]'),3.90,0),
    ('Aspartic Acid v2', Chem.MolFromSmarts('[$([CX3][CD2]C(N)C(=O))](=O)[OX1H0-,OX2H1]'), 3.9, 0),
    ('Arginine', Chem.MolFromSmarts('[$(N[CD2][CD2])]C(=N)N'), 12.5, 1),
    ('Lysine', Chem.MolFromSmarts('[$([ND1][CD2][CD2][CD2][CD2]C(N)C(=O))]'), 10.8, 1),
    ('End Carboxylate (avg)', Chem.MolFromSmarts('[$([CX3]CN)](=O)[OX1H0-,OX2H1]'), 3.6, 0),
    # ('End Amine (avg)',Chem.MolFromSmarts('[$([NX3;H2,H1;!$(NC=O)]CC=O)]'),9.15,1),
    ('End Amine (avg) PseudoAtom safe', Chem.MolFromSmarts(
        '[$([NX3;H2,H1;!$(NC=O);!$(N[#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193])]CC=O)]'),
     8.6, 1),
    ('Acrydine', Chem.MolFromSmarts('c1c2ccccc2nc2ccccc21'), 6.15, 1),
    ('Generic Carboxylic Acid', Chem.MolFromSmarts('[CX3](=O)[OX1H0-,OX2H1]'), 4.75, 0),
    (
    'Generic tert. Amine, not amide', Chem.MolFromSmarts('[N+&!H0&NX3,NX3;$(N(C)(C)C);!$(NC=[!#6]);!$(NC#[!#6])]'), 9.8,
    1),
    ('Generic Amine, not amide', Chem.MolFromSmarts(
        '[N+,NX3;!H0;!$(NC=[!#6]);!$(NC#[!#6])!$(N[#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193])]'),
     10., 1),
]

condensed_EMBOSS = [
    ('Condensed Tyrosine', Chem.MolFromSmarts('[#189]'), 10.1, 0),
    ('Condensed Histidine', Chem.MolFromSmarts('[#179]'), 6.5, 1),
    ('Condensed Glutamic Acid', Chem.MolFromSmarts('[#177]'), 4.4, 0),
    ('Condensed Aspartic Acid', Chem.MolFromSmarts('[#174]'), 3.9, 0),
    ('Condensed Arginine', Chem.MolFromSmarts('[#172]'), 12.5, 1),
    ('Condensed Lysine', Chem.MolFromSmarts('[#182]'), 10.8, 1),
    # ('Condensed AA end amino subst.', Chem.MolFromSmarts('[#0D1]'),9.15,1),
    # Can it somehow match up the wrong end? with a protection group??
    ('Condensed AA end amino', Chem.MolFromSmarts(
        '[#171,#172,#173,#174,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193;D1]'),
     8.6, 1),
    # ('Condensed AA end Carboxylic Subst.', Chem.MolFromSmarts('[$([#0][OH])]'),2.15,0), #All acidic/basic should be replaced with glycine so they can be removed??
    ('Condensed cys end amino', Chem.MolFromSmarts('[#175;D2]'), 8.6, 1),
    # ('Condensed AA end Carboxylic Subst.', Chem.MolFromSmarts('[$([#0][OH])]'),2.15,0), #All acidic/basic should be replaced with glycine so they can be removed??
    ('Condensed AA end Carboxylic', Chem.MolFromSmarts(
        '[$([#171,#172,#173,#174,#175,#176,#177,#178,#179,#180,#181,#182,#183,#184,#185,#186,#187,#188,#189,#190,#191,#192,#193][OH])]'),
     3.6, 0)
]

# Defaut pKa_tables
pkatable = pkatable_grimsley
# pkatable = pkatable_ipc
# pkatable = pkatable_EMBOSS

condensedtable = condensed_grimsley

def find_pKas(mol, pkatable=pkatable, condensedtable=condensedtable, debug=False, returnindices=False):
    """This function finds pKa values in a supplied molecule by matching it up to SMARTS rules in a table
    Returns the list of pKa values and a list with the charge of the group at pH levels blow the pKa.
    Example: So histidine has a +1 charge at pH levels below the pKa of ~6"""
    chargelist = []
    pkalist = []
    dummy = Chem.MolFromSmarts('[#178]')  # Used Gly instead of #0 or *, for compatibility with building blocks

    if returnindices:
        atomindices = []
        for atom in mol.GetAtoms():
            atom.SetProp('O_i', str(atom.GetIdx()))  # Save the Original Idx as a property

    # Look for condensed AA atoms with potential pKa values. Substitute with Glycine
    for s in condensedtable:
        if mol.HasSubstructMatch(s[1]):
            if debug: print
            "Match: %s" % (s[0]),
            # How many??
            substructmatches = mol.GetSubstructMatches(s[1])
            n = len(substructmatches)
            if debug: print
            "times %s" % (n)
            pkalist += [s[2]] * n
            chargelist += [s[3]] * n
            if returnindices:
                for match in substructmatches:
                    atomindices += [[int(mol.GetAtomWithIdx(idx).GetProp('O_i')) for idx in
                                     match]]  # Retrieve the original index of matches
            mol = AllChem.ReplaceSubstructs(mol, s[1], dummy)[
                0]  # Delete acidic/basic groups to prevent multiple matching

    # Look for molecular fragments with known pKa's
    for s in pkatable:
        # print s[0]
        if mol.HasSubstructMatch(s[1]):
            if debug: print
            "Match: %s" % (s[0]),
            # How many??
            substructmatches = mol.GetSubstructMatches(s[1])
            n = len(substructmatches)
            if debug: print
            "times %s" % (n)
            if type(s[2]) == type(float()):
                pkalist += [s[2]] * n
                chargelist += [s[3]] * n
            else:  # If there's multiple pKa values found for the molecular substructure
                for i in range(len(s[2])):
                    pkalist += [s[2][i]] * n
                    chargelist += [s[3][i]] * n
            if returnindices:
                for match in substructmatches:
                    atomindices += [
                        [int(mol.GetAtomWithIdx(idx).GetProp('O_i')) for idx in match]]  # Retrieve the original i
            mol = AllChem.ReplaceSubstructs(mol, s[1], dummy)[0]
    if returnindices:
        return pkalist, chargelist, atomindices
    return pkalist, chargelist


def charge(ph, pkalist, chargelist):
    """Jacob Tolborgs charge model where the charge is assigned from partial charges from all pKa values at the pH point"""
    chargesum = []
    abschargesum = []
    for charge, pka in zip(chargelist, pkalist):
        # print charge, pka
        if charge == 1:
            charge = 1 / (1 + 10 ** (ph - pka))
            chargesum.append(charge)
        else:
            charge = -1 / (1 + 10 ** -(ph - pka))
            chargesum.append(charge)
            abschargesum.append(abs(charge))
    return sum(chargesum), sum(abschargesum)


def pI(pkalist, chargelist):
    """Uses Jacob Tolborgs charge function and calculates pI.
    If only only acidic or basic groups are found, the pI is ill defined and returned as 42 or -42"""
    # Check if Only acidic or basic groups are present.
    if len(unique(chargelist)) == 0:  # No pKa groups present.
        pI = 7
    elif len(unique(chargelist)) == 1:  # Only one type is present
        if chargelist[0] == 0:  # Only acidic groups are present
            pI = 3
        elif chargelist[0] == 1:  # Only basic groups are present
            pI = 11
    else:
        # Find pI by simulation in the pH range 0 to 0
        chargecol = []
        for i in range(0, 1400):
            ph = i / 100.
            chargecol.append(charge(ph, pkalist, chargelist))  # Calculate charge
        pI = argmin(
            np_abs(chargecol)) / 100.  # Simply taking the smallest absolute value, and dividing the index with 100
    # print "pI %.1f"%pI
    return pI

from olorenchemengine.representations import BaseCompoundVecRepresentation
import numpy as np
from olorenchemengine.base_class import log_arguments
class calc_pI(BaseCompoundVecRepresentation):
    @log_arguments
    def __init__(self, ph=7.4, log=True, **kwargs):
        self.ph=ph
        super().__init__(names=["pI", f"Charge{self.ph}", f"TotalAbsCharge{self.ph}"], log=False, **kwargs)

    def _convert(self, smiles, y=None):
        # find pKa values and charge class
        m = Chem.MolFromSmiles(smiles)
        pkalist, chargelist = find_pKas(m)
        # Calculate pI
        pIpred = pI(pkalist, chargelist)
        return np.array([pIpred, *charge(self.ph, pkalist, chargelist)])