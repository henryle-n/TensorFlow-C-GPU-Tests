# TensorFlow-C/GPU-Tests
## Background
Out of my curiosity in the hardware world, I want to validate the statement that many ML / AI Engineers made all over the internet: Traning models on GPU is 5-10+ faster than on CPU. I set up a simple Jupyter Notebook with two kernels (run on two different venv, one utilizes only cpu, and the other utilizes GPU). Results of TensorFlow training time by utilizing CPU vs. GPU then printed out for comparison.

In my case set-up, the CPU took ~13.4 mins to complete while the GPU get the job done in ~1.6 mins. That's **8.4** times faster.

## Benefit of ML with GPU
CPU is the brain of our computing device, either a high-end server, a desktop, or a laptop. Utilizing GPU for ML brought benefits including: faster model training, and freeing up CPU resources for other tasks/ processes that GPU can't be utilized. 
## Data Set-up:
* Utilizing MNIST Dataset: http://yann.lecun.com/exdb/mnist/
    * 60,000 digit images
* Keras instruction: https://keras.io/api/datasets/mnist/
* 30 Epochs was run


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