import subprocess
import os

# 获取显卡信息并解析
def find_max_gpu(file_path):
    '''
    find and set gpu to max memory
    '''
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        text=True
    )

    # 将输出拆分为行
    gpu_info = result.stdout.strip().split('\n')

    # 解析每行数据并找到剩余显存最大的GPU ID
    max_mem_gpu_id = max(gpu_info, key=lambda x: int(x.split(',')[1])).split(',')[0]

    os.environ['CUDA_VISIBLE_DEVICES'] = max_mem_gpu_id
    print(f"Set to use GPU with the most free memory: {max_mem_gpu_id}")

    return max_mem_gpu_id