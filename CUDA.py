# Henry Le on Jun 2020 to run with Python 3        

import pycuda
# import pycuda.driver as cdr
from pycuda import driver as cdr

cdr.init()

print ('Author : Henry Le')
print (("=")*40)

if cdr.Device.count() > 1:
    dev = 'devices'
else:
    dev = 'device'
    
print (f'Detected :: ({cdr.Device.count()}) CUDA capable {dev}')

for i in range(cdr.Device.count()):
    
    gpu_device = cdr.Device(i)  
    print (f'Device [{i}] :: {gpu_device.name()}')
    
    compute_capability = float('%d.%d' % gpu_device.compute_capability())
    print (f'\t Compute Capability: {compute_capability}')
    # in binary, 1 MB = 2^20 Bytes, in SI, simply 1 MB = 10^6 Bytes
    print (f'\t Total Memory: {gpu_device.total_memory()//(2**20)} GB')
    
  
    device_attr_tuples = gpu_device.get_attributes().items() 
    device_attr = {}
    
    for k, v in device_attr_tuples:
        device_attr[str(k)] = v
    
    num_mp = device_attr['MULTIPROCESSOR_COUNT']
    
    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    # Devices with the same major revision number are of the same core architecture. 
    # The major revision number is:
        # 8 for devices based on the NVIDIA Ampere GPU architecture
        # 7 for devices based on the Volta architecture
        # 6 for devices based on the Pascal architecture
        # 5 for devices based on the Maxwell architecture
        # 3 for devices based on the Kepler architecture
        # 2 for devices based on the Fermi architecture
        # 1 for devices based on the Tesla architecture.
    
    
    # extract the cuda core per processor based on retrieved compute_capability
    def cuda_per_sm(cap_maj, cap_min):
        return {
            # Tesla
            (1, 0):   8,      # SM 1.0
            (1, 1):   8,      # SM 1.1
            (1, 2):   8,      # SM 1.2
            (1, 3):   8,      # SM 1.3
            
            # Fermi
            (2, 0):  32,      # SM 2.0: GF100 class
            (2, 1):  48,      # SM 2.1: GF10x class
            
            # Kepler
            (3, 0): 192,      # SM 3.0: GK10x class
            (3, 2): 192,      # SM 3.2: GK10x class
            (3, 5): 192,      # SM 3.5: GK11x class
            (3, 7): 192,      # SM 3.7: GK21x class
            
            # Maxwell
            (5, 0): 128,      # SM 5.0: GM10x class
            (5, 2): 128,      # SM 5.2: GM20x class
            (5, 3): 128,      # SM 5.3: GM20x class
            
            # Pascal
            (6, 0):  64,      # SM 6.0: GP100 class
            (6, 1): 128,      # SM 6.1: GP10x class
            (6, 2): 128,      # SM 6.2: GP10x class
            
            # Volta
            (7, 0):  64,      # SM 7.0: GV100 class
            (7, 2):  64,      # SM 7.2: GV11b class
            
            # Turing
            (7, 5):  64,      # SM 7.5: TU10x class
            
        }.get((cap_maj, cap_min), 0)   # unknown architecture : return ) to flag for user

    cuda_cores_per_sm = cuda_per_sm(device_attr['COMPUTE_CAPABILITY_MAJOR'], device_attr['COMPUTE_CAPABILITY_MINOR'])
  
    print (f'\t Streaming Processors: {num_mp}\n\t CUDA Cores per Streaming Processor: {cuda_cores_per_sm}\n\t Total CUDA Cores: {num_mp*cuda_cores_per_sm}')
    
    # remove MULTIPROCESSOR_COUNT to print the rest of attributes, no need for repeat
    device_attr.pop('MULTIPROCESSOR_COUNT')
    
    for k, v in device_attr.items():
        print (f'\t {k}: {v}')
        
