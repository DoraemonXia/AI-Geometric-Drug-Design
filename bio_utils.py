import numpy as np
import pandas as pd
            
from rdkit import Chem
from rdkit.Chem import AllChem

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