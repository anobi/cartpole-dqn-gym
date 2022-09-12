from enum import Enum
from torch import cuda, backends

class DeviceUse(Enum):
    DEVICE = 1
    RNG = 2

def get_device_family(use_case=DeviceUse.DEVICE):
    device_family = 'cpu'
    cuda_available = cuda.is_available() and backends.cuda.is_built()
    mps_available = backends.mps.is_available() and backends.mps.is_built()

    if cuda_available:
        device_family = 'cuda'
    # Torch RNG Generator doesn't support MPS yet
    elif mps_available and use_case is DeviceUse.DEVICE:
        device_family = 'mps'
    
    return device_family