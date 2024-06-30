#!/bin/bash

#First: read the seq_id from .fasta file
# 检查是否提供了文件路径
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_fasta_file>"
  exit 1
fi

fasta_file=$1

# 使用awk提取FASTA文件中的序列ID
sequence_ids=$(awk '/^>/ {print substr($0, 2)}' "$fasta_file")

# 将序列ID存储到数组中
sequence_id_array=($sequence_ids)

# 输出序列ID
for id in "${sequence_id_array[@]}"; do
  echo "$id"
done