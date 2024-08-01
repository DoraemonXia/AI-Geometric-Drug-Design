"""
读取小分子，生成多个构象，进行聚类，RMSD对齐计算，最终筛选与原构象超过1.5A距离差值的分子，最后存到不同的文件夹。

"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import numpy as np

# 读取sdf文件
sdf_file = 'PC4.sdf'
suppl = Chem.SDMolSupplier(sdf_file)
mol = next(suppl)

# 生成500个构象
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=500)

# 计算所有构象之间的RMSD矩阵
rmsd_matrix = []
for conf_id1 in conf_ids:
    row = []
    for conf_id2 in conf_ids:
        rmsd = AllChem.GetBestRMS(mol, mol, prbId=conf_id1, refId=conf_id2)
        row.append(rmsd)
    rmsd_matrix.append(row)


"""
进行聚类
"""
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

# 假设rmsd_matrix是一个二维数组，表示所有构象之间的RMSD值
rmsd_matrix = np.array(rmsd_matrix)

# 使用层次聚类方法进行聚类
Z = linkage(rmsd_matrix, method='average')  # 可以选择不同的聚类方法，如'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'

# 设定聚类数目
num_clusters = 5  # 可以根据需要调整聚类数目

# 根据聚类树Z和指定的聚类数目进行划分
clusters = fcluster(Z, num_clusters, criterion='maxclust')

# 输出每个构象所属的簇编号
print("Cluster assignments:", clusters)


"""
筛掉相差过大的。
"""

from rdkit import Chem
from rdkit.Chem import AllChem


# 筛选出RMSD小于等于1.5A的构象
filtered_conf_ids = []
for conf_id in conf_ids:
    rmsd = AllChem.GetBestRMS(mol, mol, prbId=conf_id, refId=0)
    if rmsd <= 1.5:
        filtered_conf_ids.append(conf_id)

print("Filtered conformation IDs:", filtered_conf_ids)


"""
保存到不同文件夹。
"""

# 创建五个文件夹来存储不同聚类结果的小分子
import os
num_clusters = 5
cluster_folders = ['Cluster_{}'.format(i+1) for i in range(num_clusters)]

# 创建文件夹
for folder in cluster_folders:
    os.makedirs(folder, exist_ok=True)

# 将小分子保存到对应的文件夹中
for molecule_id, cluster_id in enumerate(clusters):
    if molecule_id not in filtered_conf_ids:
        # 假设这里是保存SDF文件的代码，这里使用简单的示例
        sdf_filename = 'conf_{}.sdf'.format(molecule_id)
        cluster_folder = cluster_folders[cluster_id - 1]  # cluster_id从1开始，而列表索引从0开始
        sdf_filepath = os.path.join(cluster_folder, sdf_filename)
        
        sdf_block = Chem.MolToMolBlock(mol, confId=molecule_id)
        
        # 假设这里是保存SDF文件的操作，这里只是示例，具体操作取决于你的数据格式和存储方式
        with open(sdf_filepath, 'w') as f:
            f.write(sdf_block)
