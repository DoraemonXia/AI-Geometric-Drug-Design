import gzip
import numpy as np
import pandas as pd
from typing import Collection, Optional, Literal

from rdkit import Chem
from rdkit.Chem import AllChem

from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence
import biotite.structure as struc
from Bio import SeqIO



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



def get_fasta_ids_and_sequences(fasta_file):
    ids, sequences = [], []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    return ids, sequences

"""
Example Usage
fasta_file = 'path_to_your_fasta_file.fasta'
sequence_ids, sequences = get_fasta_ids_and_sequences(fasta_file)
print(sequence_ids)
print(sequences)
"""



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


def get_pdb_file_information( file_name, if_multi_struc = False, atoms = ["N", "CA", "C", "O"] ):
    """
    Get PDB seq and backbone atom from .pdb file
    """
    structure = PDBFile.read(file_name)
    backbone_list = []
    pdb_seq_list = []
    
    for i in range(len(structure.get_structure())):
        chain = structure.get_structure()[i]  #There has a problem about multi model or multi chain

        # backbone = chain[struc.filter_backbone(chain)]   #this is backbone including N, CA, C atom
        
        amino_acids_atoms = chain[struc.filter_amino_acids(chain)]  #filter atom in amino acids
        
        selected_atoms = [c for c in amino_acids_atoms if c.atom_name in atoms] #this is backbone including N, CA, C, and O atom
        
        amino_acids = [ProteinSequence.convert_letter_3to1( selected_atoms[len(atoms)*i].res_name ) for i in range(int(len(selected_atoms)/len(atoms)))]
    
        pdb_seq = ""
        for j in amino_acids:
            pdb_seq+=j
            
        backbone_list.append(selected_atoms)
        pdb_seq_list.append(pdb_seq)
            
    if if_multi_struc:
        return backbone_list, pdb_seq_list
    else:
        return backbone_list[0], pdb_seq_list[0]
 
'''
# Example usage:
file_name = "../../../Workspace/Combs/data/LigandMPNN/Protein_Metal/1dwh.pdb"
backbone, pdb_seq = get_pdb_file_information(file_name)
backbone_coord = np.array( [ backbone[i].coord for i in range(len(backbone)) ] )
'''

def write_coords_to_pdb(coords: np.ndarray, out_fname: str, if_bonds = False, select_atoms = ["N", "CA", "C", "O"] ) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % len(select_atoms) == 0, f"Expected "+str(len(select_atoms))+"N coords, got {len(coords)}"
    atoms = []
    for i, (n_coord, ca_coord, c_coord, o_coord) in enumerate(
        (coords[j : j + len(select_atoms)] for j in range(0, len(coords), len(select_atoms) ))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * len(select_atoms) + 1,
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
            atom_id=i * len(select_atoms) + 2,
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
            atom_id=i * len(select_atoms) + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom4 = struc.Atom(
            o_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * len(select_atoms) + 4,
            res_name="GLY",
            atom_name="O",
            element="O",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3, atom4])
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
origin_seq = []
file_list = []
for i in range( 1500 ):
    try:
        backbone, pdb_seq = get_pdb_file_information( files[i] )
        #backbone_coord = extract_backbone_coords(files[i])
        assert len(backbone)==4*len(pdb_seq)
        backbone_coord = np.array( [ backbone[i].coord for i in range(len(backbone)) ] )
        write_coords_to_pdb( backbone_coord, "backbone/"+all_files_and_folders[i] )
        origin_seq.append( pdb_seq )
        file_list.append(all_files_and_folders[i])
    except:
        print(all_files_and_folders[i])
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




from Bio import SeqIO
import os

def split_fasta(input_fasta, output_dir, chunk_size=100):
    """
    Split fasta files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录

    base_name = os.path.basename(input_fasta)
    name, ext = os.path.splitext(base_name)

    records = list(SeqIO.parse(input_fasta, "fasta"))

    for i in range(0, len(records), chunk_size):
        chunk_records = records[i:i+chunk_size]
        output_file = os.path.join(output_dir, f"{name}_{i//chunk_size}{ext}")
        SeqIO.write(chunk_records, output_file, "fasta")
        print(f"Saved {len(chunk_records)} records to {output_file}")

'''
# Example usage:
input_fasta = "output.fasta"  # 输入的FASTA文件路径
output_dir = "output/"  # 输出目录
split_fasta(input_fasta, output_dir)
'''


def parse_RNAfold(file_path):
    '''
    Transfer a RNA secondary structure file into secondary structure dict. 
    '''
    sequences_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('>'):
                seq_name = line[1:]
                i += 2
                sequence = ''
                while i < len(lines) and not lines[i].startswith('>'):
                    sequence += lines[i].split()[0]
                    i += 1
                sequences_dict[seq_name] = sequence
            else:
                i += 1
    return sequences_dict

'''
# Example usage:
Robin_path = 'Robin/RNAfold.out'  # replace with your filepath
Robin_RNA_ss = parse_RNAfold(Robin_path)
'''


def parse_rna_structure(structure):
    edges = []
    stack = []
    for i, symbol in enumerate(structure):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')':
            if stack:
                opening_bracket_index = stack.pop()
                edges.append((opening_bracket_index, i))
                edges.append((i, opening_bracket_index))
    
    for i in range(len(structure)-1):
        edges.append( (i, i+1) )
        edges.append( (i+1, i) )
    return edges

'''
# Example usage:
Robin_RNA_Graph = {}
for i in Robin_sequences.keys():
    x = torch.tensor( Robin_RNA_FM_repre[i] , dtype=torch.float)  #modify torch.long=>torch.float  #node features
    edge_index = torch.tensor(np.array(parse_rna_structure(Robin_RNA_ss[i]) ).T, dtype=torch.long)  #transfer rna_structure into edge_index
    data = Data(x=x, edge_index=edge_index)
    Robin_RNA_Graph[i]=data
'''

from rdkit import Chem

def load_multiple_mol2(file_path):
    """
    Load a MOL2 file containing multiple molecules and return a list of molecule objects.
    
    Parameters:
    - file_path (str): Path to the MOL2 file.
    
    Returns:
    - mol_list (list of rdkit.Chem.rdchem.Mol): List of molecule objects.
    """
    mol_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    mol_block = []
    for line in lines:
        if line.startswith('@<TRIPOS>MOLECULE'):
            if mol_block:
                mol = Chem.MolFromMol2Block(''.join(mol_block), sanitize=False)
                if mol is not None:
                    mol_list.append(mol)
            mol_block = [line]
        else:
            mol_block.append(line)
    
    # Add the last molecule
    if mol_block:
        mol = Chem.MolFromMol2Block(''.join(mol_block), sanitize=False)
        if mol is not None:
            mol_list.append(mol)
    
    if not mol_list:
        raise ValueError(f"Failed to load any molecules from {file_path}")
    
    return mol_list

'''
# Example usage:
small_molecules = load_multiple_mol2("RLDock/RLDock/mol_0/_cluster.mol2")
'''


def load_rna_bases_from_mol2(file_path):
    """
    Load RNA bases from a MOL2 file and return a list of atom indices for each base.
    
    Parameters:
    - file_path (str): Path to the MOL2 file.
    
    Returns:
    - base_atoms (list of list of int): List of atom indices for each base in the RNA.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    base_atoms = {}
    
    in_atom_section = False

    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
        elif line.startswith("@<TRIPOS>"):
            in_atom_section = False
        elif in_atom_section:
            tokens = line.split()
            atom_index = int(tokens[0]) - 1  # Atom index starts from 1 in MOL2
            base_index = int(tokens[6])      # Base index from the 8th column

            if base_index not in base_atoms:
                base_atoms[base_index] = []
            
            base_atoms[base_index].append(atom_index)
    
    return [base_atoms[key] for key in sorted(base_atoms.keys())]

'''
# Example usage:
rna_base_atoms = load_rna_bases_from_mol2(rna_file_path)
'''

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

def smiles_to_png(smiles, output_file):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    img = Draw.MolToImage(mol, size=(300, 300), kekulize=True, wedgeBonds=True, includeAtomNumbers=False)
    img = img.convert("RGBA")

    datas = img.getdata()
    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))  # 将白色背景设置为透明
        else:
            new_data.append(item)
    img.putdata(new_data)

    img.save(output_file, "PNG")

'''
# Example usage:
smiles = "CN1C(=O)N(C)c2nc[nH]c2C1=O"
output_file = "1eht.png"
smiles_to_png(smiles, output_file)
'''


from rdkit import Chem
from rdkit.Chem import AllChem

def save_multiple_conformers_to_sdf(smiles, num_confs=100, output_file='output.sdf'):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDG()
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    
    # 优化每个构象
    results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=4)

    writer = Chem.SDWriter(output_file)
    for conf_id, (energy, _) in zip(conf_ids, results):
        mol.SetProp("_Name", f'Conformer_{conf_id}_Energy_{energy}')
        writer.write(mol, confId=conf_id)
        print(f"Conformer ID: {conf_id}, Energy: {energy}")
    writer.close()

'''
# Example usage:
smiles = "Cc1ccc(CNC(c2nccn2C)c2ccc(F)cc2)cc1C"  # 示例SMILES
save_multiple_conformers_to_sdf(smiles, num_confs=100, output_file='mol_0.sdf')
'''

from rdkit import Chem
from rdkit.Chem import AllChem

def translate_molecule(sdf_file, output_file, dx, dy, dz):
    # read sdf files
    suppl = Chem.SDMolSupplier(sdf_file)
    writer = Chem.SDWriter(output_file)
    
    for mol in suppl:
        if mol is None:
            continue
        # get all coordinates from mol.
        conf = mol.GetConformer()
        for atom_idx in range(mol.GetNumAtoms()):
            pos = list(conf.GetAtomPosition(atom_idx))
            # move coords
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz
            conf.SetAtomPosition(atom_idx, pos)
        # write in new files.
        writer.write(mol)
    writer.close()

'''
# Example usage:
translate_molecule('mol/mol_0.sdf', 'mol/output.sdf', -4, 15, 15)
'''



import os
def convert_pdb_to_sdf(input_folder):
    '''
    Leverage obabel to transform mol.pdb to sdf.
    '''
    failed_conversions = []  # 存储转换失败的文件名

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否以 .pdb 结尾
        if filename.endswith(".pdb"):
            pdb_file = os.path.join(input_folder, filename)
            sdf_file = os.path.join(input_folder, filename.replace(".pdb", ".sdf"))
            
            # 构建 Open Babel 命令
            command = f"obabel -i pdb {pdb_file} -o sdf -O {sdf_file}"
            
            # 执行命令并检查返回值
            result = os.system(command)
            if result != 0:
                # 如果命令执行失败，记录文件名
                failed_conversions.append(filename)
                print(f"Failed to convert {pdb_file}")
            else:
                print(f"Converted {pdb_file} to {sdf_file}")

    return failed_conversions

'''
# Example usage:
input_folder = "source_data/Hariboss/mol/"  # 替换为你的 PDB 文件所在的文件夹
failed_files = convert_pdb_to_sdf(input_folder)
'''


'''
Then we will display how to get ligand from .cif file and transform it into pdb.
I think its' mean just use obabel to add bonds.
'''

from rdkit.Chem import rdmolfiles
def save_mol_to_pdb(mol, filename):
    """
    将 RDKit 的 mol 对象保存为 PDB 格式文件。

    :param mol: RDKit 的 mol 对象
    :param filename: 保存的文件名（包括路径）
    """
    if mol is None:
        raise ValueError("Invalid molecule object.")
    
    # 保存为 PDB 文件
    with open(filename, 'w') as pdb_file:
        pdb_block = Chem.MolToPDBBlock(mol)
        pdb_file.write(pdb_block)


from Bio.PDB import MMCIFParser

def extract_ligand_info_from_cif(cif_file, ligand_name, chain_id="A" ):
    """
    从 CIF 文件中提取指定配体的原子信息。
    
    参数:
    - cif_file: CIF 文件的路径
    - ligand_name: 配体的名称
    
    返回:
    - ligand_info: 一个包含配体原子信息的字典
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)[0]

    ligand_info = {}
    atoms = []
    for chain in structure:
        if chain.id!=chain_id:
            continue
        not_found_residue = True
        for residue in chain:
            #print(residue.resname)
            if residue.resname not in ['A', 'C', 'G', 'U', 'T']:
                if residue.resname == ligand_name and not_found_residue:
                    not_found_residue = False
                    for atom in residue.get_atoms():
                        atom_name = atom.get_id()  # 提取原子名
                        residue_name = residue.get_resname()  # 提取残基名
                        x, y, z = atom.get_coord()  # 提取坐标
                        atoms.append((atom_name, residue_name, x, y, z))
                    
                        
    mol = Chem.RWMol()
    
    # add atoms
    atom_indices = {}
    for idx, (atom_name, residue_name, x, y, z) in enumerate(atoms):
        atom_idx = mol.AddAtom(Chem.Atom(atom_name[0]))
        atom_indices[idx] = atom_idx
    
    # 添加坐标
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (atom_name, residue_name, x, y, z) in enumerate(atoms):
        conf.SetAtomPosition(atom_indices[idx], (float(x), float(y), float(z)))
    
    mol.AddConformer(conf)
    return mol

'''
Example Usage
for i in range(len(rna_list)):
    cif_file = "/home/raojh/datasets/HARIBOSS/cifs/"+rna_list[i]+".cif"
    mol = extract_ligand_info_from_cif(cif_file, smile_name[i], smile_chain[i] )
    save_mol_to_pdb(mol, 'source_data/Hariboss/mol/'+rna_list[i]+'_mol_'+str(i)+'.pdb')
'''


#read atom in a chain from cif and save it into pdb
from Bio.PDB import MMCIFParser, PDBIO, Select

def extract_rna_atoms_from_cif(cif_file, chain_id=-1):
    """
    从 CIF 文件中提取指定链或所有链上的 RNA 原子的列表，并生成可用于创建新 PDB 文件的结构。
    
    参数:
    - cif_file: CIF 文件的路径
    - chain_id: 要提取的链的 ID。如果为 -1，提取所有链上的 RNA 原子。
    
    返回:
    - rna_atoms: 一个包含指定链上 RNA 原子的列表
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)[0]

    rna_atoms = []
    rna_bases = {"A", "U", "C", "G"}  # RNA 的四种碱基

    for chain in structure:
        if chain_id != -1 and chain.id != chain_id:
            continue

        for residue in chain:
            if residue.resname in rna_bases:
                for atom in residue.get_atoms():
                    rna_atoms.append(atom)
    
    return rna_atoms

def save_atoms_to_pdb_simple(atoms, output_file):
    """
    将提取的 RNA 原子保存为 PDB 文件（简化版）。
    
    参数:
    - atoms: 包含 RNA 原子的列表
    - output_file: 保存的 PDB 文件路径
    """
    with open(output_file, 'w') as pdb_file:
        for i, atom in enumerate(atoms, start=1):
            atom_line = (
                f"ATOM  {i:5d} {atom.get_name():>4} {atom.get_parent().get_resname():>3} "
                f"{atom.get_parent().get_parent().id:>1}{atom.get_parent().id[1]:>4}    "
                f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                f"  1.00  0.00           {atom.element:>2}\n"
            )
            pdb_file.write(atom_line)
        pdb_file.write("END\n")

'''
Example Usage
cif_file = "/home/raojh/datasets/HARIBOSS/cifs/8hba.cif"
rna_atoms = extract_rna_atoms_from_cif(cif_file, chain_id=-1)
output_pdb_file = "rna_chain_A.pdb"
save_atoms_to_pdb_simple(rna_atoms, output_pdb_file)

extract_rna_atoms_from_cif  这个有时候会出现读不到原子的情况,所以需要重新下载pdb,然后重新读pdb

'''


import os
import requests
def download_pdb_files(pdb_ids, output_folder, timeout=600):
    """
    批量下载指定 PDB ID 的 PDB 文件到指定文件夹。如果PDB文件下载失败，则尝试下载CIF文件。

    参数:
    - pdb_ids (list of str): PDB ID 列表。
    - output_folder (str): 下载的 PDB 文件保存的文件夹路径。
    - timeout (int): 下载等待时间（秒），默认为600秒。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdb_base_url = "https://files.rcsb.org/download/"
    cif_base_url = "https://files.rcsb.org/download/"

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.upper()
        pdb_url = f"{pdb_base_url}{pdb_id}.pdb"
        cif_url = f"{cif_base_url}{pdb_id}.cif"

        try:
            response = requests.get(pdb_url, timeout=timeout)

            if response.status_code == 200:
                file_path = os.path.join(output_folder, f"{pdb_id}.pdb")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"{pdb_id}.pdb 已下载并保存到 {file_path}")
            else:
                print(f"下载 {pdb_id}.pdb 失败，状态码: {response.status_code}")
                # 尝试下载 CIF 文件
                download_cif_file(cif_url, pdb_id, output_folder, timeout)

        except requests.exceptions.Timeout:
            print(f"下载 {pdb_id}.pdb 超时，尝试下载 CIF 文件。")
            download_cif_file(cif_url, pdb_id, output_folder, timeout)
        except requests.exceptions.RequestException as e:
            print(f"下载 {pdb_id}.pdb 出现错误: {e}，尝试下载 CIF 文件。")
            download_cif_file(cif_url, pdb_id, output_folder, timeout)

def download_cif_file(url, pdb_id, output_folder, timeout):
    """
    尝试下载CIF文件。

    参数:
    - url (str): CIF 文件的下载链接。
    - pdb_id (str): PDB ID，用于命名文件。
    - output_folder (str): 下载的文件保存的文件夹路径。
    - timeout (int): 下载等待时间（秒）。
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            file_path = os.path.join(output_folder, f"{pdb_id}.cif")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"{pdb_id}.cif 已下载并保存到 {file_path}")
        else:
            print(f"下载 {pdb_id}.cif 失败，状态码: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"下载 {pdb_id}.cif 超时，跳过此文件。")
    except requests.exceptions.RequestException as e:
        print(f"下载 {pdb_id}.cif 出现错误: {e}")

'''
Example Usage
download_pdb_files(["5afi"], "source_data/Hariboss/pdb/")  #默认下载pdb,不提供pdb的情况下则下载cif
'''

#保存完RNA简化的分子后，下一步需要将其转化为需要的坐标列表
from Bio.PDB import PDBParser

def load_rna_bases_from_pdb(file_path, return_seq=False, chain_id=-1):
    """
    从PDB文件中加载RNA碱基并返回每个碱基对应的原子列表。
    
    参数:
    - file_path (str): PDB文件的路径。
    - return_seq (bool): 是否返回序列信息。如果为True，则返回序列。
    - chain_id (str/int): 要提取的链的 ID。如果为-1，提取所有链上的 RNA 原子。

    返回:
    - base_atoms (dict): 每个碱基的原子列表，键为碱基ID（序号+碱基名，如 17_G），值为原子坐标的列表。
    - rna_sequences (list): 如果return_seq为True，则返回序列信息列表。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", file_path)

    base_atoms = {}
    rna_sequences = []  # 存储每条链的序列信息
    rna_bases = {"A", "U", "C", "G"}  # RNA 的四种碱基
    index = 1  # 用于生成序号

    for model in structure:
        for chain in model:
            if chain_id != -1 and chain.id != chain_id:
                continue

            chain_seq = []
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in rna_bases:
                    base_id = f"{index}_{resname}"  # 生成序号+碱基名的ID
                    atoms = [atom.get_coord() for atom in residue]
                    base_atoms[base_id] = atoms
                    if return_seq:
                        chain_seq.append(resname)
                    index += 1

            if return_seq and chain_seq:
                rna_sequences.append("".join(chain_seq))

    if return_seq:
        return base_atoms, rna_sequences
    else:
        return base_atoms

#配套对应的
def convert_base_atoms_to_matrix(base_atoms):
    # 对每个碱基，提取原子坐标并将其转换为矩阵
    base_matrices = {}
    for base_key, atom_coords in base_atoms.items():
        # 将 atom_coords 列表转换为 n*3 的矩阵
        base_matrix = np.array(atom_coords)
        base_matrices[base_key] = base_matrix
    return base_matrices

#对应计算其接触矩阵的函数
import numpy as np
def calculate_min_distance_matrix(base_atoms, mol_coords):
    # 初始化距离矩阵
    num_bases = len(base_atoms)
    num_mol_atoms = len(mol_coords)
    distance_matrix = np.zeros((num_bases, num_mol_atoms))
    
    # 遍历每个碱基
    for i, (base_key, base_coords) in enumerate(base_atoms.items()):
        # 遍历分子中的每个原子
        for j, mol_coord in enumerate(mol_coords):
            # 计算分子原子与当前碱基中所有原子的距离
            distances = np.linalg.norm(base_coords - mol_coord, axis=1)
            # 取最短距离
            min_distance = np.min(distances)
            distance_matrix[i, j] = min_distance
    
    return distance_matrix

'''
Example Usage

rna_seq_list = []
pairwise_list = []

for i in range(len(rna_list)):
    pdb_file = "source_data/Hariboss/rna/"+rna_list[i]+"_"+rna_chain[i]+".pdb"
    base_atoms, rna_seq = load_rna_bases_from_pdb(pdb_file, return_seq=True, chain_id=rna_chain[i])

    rna_seq_list.append(rna_seq[0])
    supplier = Chem.SDMolSupplier("source_data/Hariboss/mol/"+rna_list[i]+"_mol_"+str(i)+".sdf",sanitize=False)
    for mol in supplier:
        coords = []
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coords.append({"atom_id": atom.GetIdx(), "position": (pos.x, pos.y, pos.z)})

    base_matrices = convert_base_atoms_to_matrix(base_atoms)
    coord_matrix = convert_atom_dict_to_matrix(coords)

    distance_matrix = calculate_min_distance_matrix(base_matrices, coord_matrix)
    
    pairwise_list.append(distance_matrix)
'''


def extract_atom_coordinates(pdb_file, atoms_to_extract = ['C4\''] ):
    '''
    Extract the atoms which you want from pdb files.
    '''
    #atoms_to_extract = ['C4\'', 'N1', 'P']
    #atoms_to_extract = ['C4\'']
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    coordinates_list = []
    atom_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    atoms = {atom.name: atom for atom in residue.get_atoms()}
                    coordinates = [atoms[atom_name].get_coord() for atom_name in atoms if atom_name in atoms_to_extract]
                    atom_name = [atoms[atom_name].name for atom_name in atoms if atom_name in atoms_to_extract]
                    atom_list.extend( [RNA_atom_id[i] for i in atom_name ] )
                    coordinates_list.extend(coordinates)
    return np.array(atom_list),np.array(coordinates_list)

'''
Example Usage
C4_coords = extract_atom_coordinates(pdb_file_path)
'''


import os

def convert_pdb_to_sdf(input_folder):
    '''
    Transform .pdb files into .sdf files in a specific folder.
    '''
    failed_conversions = []  # 存储转换失败的文件名

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否以 .pdb 结尾
        if filename.endswith(".pdb"):
            pdb_file = os.path.join(input_folder, filename)
            sdf_file = os.path.join(input_folder, filename.replace(".pdb", ".sdf"))
            
            # 构建 Open Babel 命令
            command = f"obabel -i pdb {pdb_file} -o sdf -O {sdf_file} -d"   # -d is used to remove H atoms.
            
            # 执行命令并检查返回值
            result = os.system(command)
            if result != 0:
                # 如果命令执行失败，记录文件名
                failed_conversions.append(filename)
                print(f"Failed to convert {pdb_file}")
            else:
                print(f"Converted {pdb_file} to {sdf_file}")

    return failed_conversions

'''
Example Usage
input_folder = "source_data/Hariboss/mol/"  # 替换为你的 PDB 文件所在的文件夹
failed_files = convert_pdb_to_sdf(input_folder)
'''
