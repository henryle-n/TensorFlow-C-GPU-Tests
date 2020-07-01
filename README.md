# TensorFlow-C/GPU-Benchmark
## Background
Out of my curiosity for hardware speed in the world of Machine Learning (ML), I want to validate the statement that many ML / AI Engineers made all over the internet: Traning models on GPU could be 5-10+ times faster than on CPU (pending on CPU/ GPU models). 

So in this mini-project, I set up a simple Jupyter Notebook with two kernels (run on two different venv, one utilizes only cpu, and the other utilizes only GPU). TensorFlow training time by utilizing CPU vs. GPU is then clocked and printed out for comparison.

<br>

![i7 vs. rtx 2070 Super](Img/readme.png)


## Observation
* Model trained time with:
    * CPU :: 13.4 minutes
    * GPU :: 1.6 minutes

* The difference is :: GPU computed 8.4 times faster than CPU
* Note: ***.Adam()*** optimization package was used in both cases
## Benefit of ML by GPU
CPU is the brain of our computing devices, no matter if it is a high-end server, a desktop, or a laptop, or just a simple tablet. Utilizing GPU for ML brought benefits including: 
* Faster model training
* Freeing up CPU resources for other tasks/ processes that GPU can't be utilized

## Differences between GPU and CPU:
* CPU has fewer cores (4 x 2 with i7-7700K) which run processes sequentially, a few threads at a time
* GPU has more cores (40 x 64 with RTX 2070 SUPER) which allow parallel computing with thousands of threads at a time
* In deep learning, the host code runs on CPU where as CUDA code runs on GPU
* GPU is bandwidth optimized (carry multiple large size packages, in trade-off for speed)
* CPU is latency (i.e. memory/RAM access time, high speed but fewer and smaller packages) optimized
* Bandwidth of GPU is significantly larger than that of CPU, thanks to the VRAM
* The GPU capability of *Thread Parallelism* in GPU surpasses the latency

## Data Set-up:
* Utilizing MNIST Dataset: http://yann.lecun.com/exdb/mnist/
    * 60,000 digit images
* Keras instruction: https://keras.io/api/datasets/mnist/
* 30 Epochs were run for comparison

## Hardware:
* Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
	* Base speed:	4.20 GHz
	* Full speed:	4.49 GHz (on-load Turbo Boost)
	* Sockets:	1
	* Cores:	4
    * Bus rate	4 × 8 GT/s
	* Logical processors:	8
	* Virtualization:	Enabled
	* L1 cache:	256 KB
	* L2 cache:	1.0 MB
	* L3 cache:	8.0 MB
    * Full Specs: <a href="https://ark.intel.com/content/www/us/en/dark/products/97129/intel-core-i7-7700k-processor-8m-cache-up-to-4-50-ghz.html">Intel Website</a>

* MSI NVIDIA GeForce RTX 2070 SUPER GAMING X
    * BIOS Version: 90.04.86.00.62
    * Stream Processors:	2560 CUDA Cores
    * Interface:	PCI Express 3.0 x16
    * Supported APIs:	DirectX: 12 | OpenGL: 4.5
    * Memory Speed:	14 Gb/s
    * Memory Configuration:	8 GB
    * Memory Interface:	GDDR6
    * Memory Interface Width:	256-Bit
    * Memory Bandwidth:	448 GB/s
    * Full Specs: <a href="https://www.msi.com/Graphics-card/GeForce-RTX-2070-GAMING-X-8G/Specification">MSI Website</a>

