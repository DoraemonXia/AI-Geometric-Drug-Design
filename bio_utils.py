import gzip
import numpy as np
import pandas as pd
from typing import Collection, Optional, Literal

from rdkit import Chem
from rdkit.Chem import AllChem

from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence
import biotite.structure as struc




from biotite.structure import AtomArray, filter_backbone,filter_amino_acids


def contains_functional_group(molecule_smiles, functional_group_smarts):
    """
    判断给定的分子是否包含指定的官能团。

    :param molecule_smiles: 分子的 SMILES 表示
    :param functional_group_smarts: 官能团的 SMARTS 表示
    :return: 如果包含该官能团，返回 True；否则返回 False
    """
    # 创建分子和官能团的分子对象
    molecule = Chem.MolFromSmiles(molecule_smiles)
    functional_group = Chem.MolFromSmarts(functional_group_smarts)
    
    # 使用 SubstructMatch 判断分子是否包含官能团
    if molecule.HasSubstructMatch(functional_group):
        return True
    else:
        return False
    

"""
how to use
"""
# count  = 0
# for i in range(len(Smiles)):
#     if contains_functional_group(Smiles[i], "[*]CC([*])(CC(=O)OCCCC)C(=O)OCCCC"):
#         count+=1
# print(count)


def generate_fasta(rna_sequences, names=None, fasta_file_path='output.fasta', reverse=False):
    """
    Generate a FASTA file from RNA sequences.

    Parameters:
    - rna_sequences (list): List of RNA sequences.
    - names (list, optional): List of names corresponding to RNA sequences. If None, default names will be used.
    - fasta_file_path (str, optional): Path to save the generated FASTA file.
    - reverse (bool, optional): If True, replace 'T' with 'U' in the sequences.

    Returns:
    - None
    """
    if not names:
        # If names are not provided, use default names (seq_0, seq_1, ...)
        names = [f"seq_{i}" for i in range(len(rna_sequences))]
    
    if len(rna_sequences) != len(names):
        raise ValueError("Number of RNA sequences must match the number of names.")

    with open(fasta_file_path, 'w') as fasta_file:
        for i in range(len(rna_sequences)):
            sequence = rna_sequences[i]
            name = names[i]
            
            if reverse:
                # If reverse is True, replace 'T' with 'U'
                sequence = sequence.replace('T', 'U')

            fasta_file.write(f">{name}\n{sequence}\n")

'''
# Example usage:
rna_sequences = ["AUGCUAGUAC", "CCGUAGCGUA", "UAGCUAGCUA"]
names = ["seq_1", "seq_2", "seq_3"]
generate_fasta(rna_sequences, names, fasta_file_path='output.fasta', reverse=True)
'''

def smiles_to_ecfp1024(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(np.array(ecfp))
            else:
                print(f"Invalid SMILES string: {smiles}")
        except Exception as e:
            print(f"Error processing SMILES string: {smiles}\nException: {e}")
            fingerprints.append(np.zeros((1, 1024), dtype=np.bool))

    fingerprint_matrix = np.vstack(fingerprints)
    return fingerprint_matrix

'''
# Example usage:
smiles_list = ["","",""]
smiles_to_ecfp1024(smiles_list)
'''


def get_pdb_file_information( file_name, if_multi_struc = False ):
    """
    from pdb file generate backbone and pdb_seq
    """
    structure = PDBFile.read(file_name)

    backbone_list = []
    pdb_seq_list = []
    
    for i in range(len(structure.get_structure())):
        chain = structure.get_structure()[i]  #There has a problem about multi model or multi chain

        backbone = chain[struc.filter_backbone(chain)]   #this is backbone including N, CA, C atom

        amino_acids = [ProteinSequence.convert_letter_3to1( backbone.res_name[3*i] ) for i in range(int(len(backbone.res_name)/3))]
        pdb_seq = ""
        for j in amino_acids:
            pdb_seq+=j
            
        backbone_list.append(backbone)
        pdb_seq_list.append(pdb_seq)
            
    if if_multi_struc:
        return backbone_list, pdb_seq_list
    else:
        return backbone_list[0], pdb_seq_list[0]
 
'''
# Example usage:
file_name = "../../../Workspace/Combs/data/LigandMPNN/Protein_Metal/1dwh.pdb"
backbone, pdb_seq = get_pdb_file_information(file_name)
'''

import numpy as np

def write_coords_to_pdb(coords: np.ndarray, out_fname: str, if_bonds = False) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % 3 == 0, f"Expected 3N coords, got {len(coords)}"
    atoms = []
    for i, (n_coord, ca_coord, c_coord) in enumerate(
        (coords[j : j + 3] for j in range(0, len(coords), 3))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 1,
            res_name="GLY",
            atom_name="N",
            element="N",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3])
    full_structure = struc.array(atoms)

    # Add bonds
    if if_bonds:
        full_structure.bonds = struc.BondList(full_structure.array_length())
        indices = list(range(full_structure.array_length()))
        for a, b in zip(indices[:-1], indices[1:]):
            full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

    # Annotate secondary structure using CA coordinates
    # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
    # https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # a = alpha helix, b = beta sheet, c = coil
    # ss = struc.annotate_sse(full_structure, "A")
    # full_structure.set_annotation("secondary_structure_psea", ss)

    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname

'''
# Example usage:
backbone_coord = np.array( [ backbone[i].coord for i in range(len(backbone)) ] )
write_coords_to_pdb( backbone_coord, "test_0.pdb" )
'''

import gzip
from typing import Collection, Optional, Literal
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure import AtomArray, filter_backbone,filter_amino_acids

def extract_backbone_coords(
    fname: str, atoms: Collection[Literal["N", "CA", "C", "O"]] = ["N", "CA", "C", "O"]
) -> Optional[np.ndarray]:
    """Extract the coordinates of specified backbone atoms"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        structure = PDBFile.read(f)
    # if structure.get_model_count() > 1:
    #     return None
    chain = structure.get_structure()[0]
    #backbone = chain
    amino_acids = chain[filter_amino_acids(chain)]  #filter atom in amino acids
    selected_atoms = [c for c in amino_acids if c.atom_name in atoms]
    coords = np.vstack([c.coord for c in selected_atoms])
    return coords