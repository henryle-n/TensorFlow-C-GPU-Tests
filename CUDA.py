# Henry Le on Jun 2020 to run with Python 3        

import pycuda
# import pycuda.driver as cdr
from pycuda import driver as cdr

cdr.init()

print ('Author : Henry Le')
print (("=")*40)
print ('Detected {} CUDA Capable device(s)'.format(cdr.Device.count()))

for i in range(cdr.Device.count()):
    
    gpu_device = cdr.Device(i)
    print ('Device {}: {}'.format( i, gpu_device.name()))
    compute_capability = float( '%d.%d' % gpu_device.compute_capability() )
    print ('\t Compute Capability: {}'.format(compute_capability))
    print ('\t Total Memory: {} megabytes'.format(gpu_device.total_memory()//(1024**2)))
    
  
    device_attr_tuples = gpu_device.get_attributes().items() 
    device_attr = {}
    
    for k, v in device_attr_tuples:
        device_attr[str(k)] = v
    print("Device Attributes : ", "\n", ("-")*30, "\n", device_attr)
    
    num_mp = device_attr['MULTIPROCESSOR_COUNT']
    
    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    # extract the cuda core per processor based on retrieved compute_capability
    cuda_cores_per_mp = { 
        5.0 : 128,
        5.2 : 128,
        5.3 : 16,
        6.0 : 128,
        6.1 : 32,
        6.2 : 16,
        7.0 : 128,
        7.2 : 16,
        7.5 : 128,
        8.0 : 128}[compute_capability]
  

    print ('\t ({}) Multiprocessors, ({}) CUDA Cores \n\t Multiprocessor: Total ({}) CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))
    
    device_attr.pop('MULTIPROCESSOR_COUNT')
    
    for k, v in device_attr.items():
        print ('\t {}: {}'.format(k, v))
        
