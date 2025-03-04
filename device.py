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


def get_available_device_slurm(num_device, only_empty=True):
    """
    Retrieves the list of available GPU indices based on the local order in the CUDA_VISIBLE_DEVICES 
    environment variable. It can filter GPUs based on memory usage, returning either all available GPUs 
    or only those with little or no memory usage.

    Parameters:
    num_device (int): The number of available devices to return.
    only_empty (bool): If True, only returns GPUs with little or no memory usage. Default is True.

    Returns:
    List[int]: A list of local GPU indices based on CUDA_VISIBLE_DEVICES.
    """
    # Get the list of GPUs specified in the CUDA_VISIBLE_DEVICES environment variable
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    
    visible_devices = [device.strip() for device in visible_devices if device.strip()]
    available_gpus = GPUtil.getGPUs()
    available_gpu_ids = [gpu.id for gpu in available_gpus if str(gpu.id) in visible_devices]

    if only_empty:
        available_gpu_ids = [gpu_id for gpu_id in available_gpu_ids if next(gpu.memoryUsed for gpu in available_gpus if gpu.id == gpu_id) < 1000]
    local_indices = [visible_devices.index(str(gpu_id)) for gpu_id in available_gpu_ids]

    return local_indices[:num_device]