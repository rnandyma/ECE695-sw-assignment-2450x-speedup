
1. Unzip the directory in a single folder only. In all the codes the benchmark files' path is same as the code path. So it is important that all the cpu-benchmark files and the codes are in the same directory.
2. Due to Blackboard submission contraint on file size, CPU benchmrak files were not submitted. So CPU benchmark files have to be generated first time if you want to run GPU code. Each layer will generate benchmark file by default name "layer_<layer_number>_<batch_size>". For example for layer 1, batch size 256 the corresponding benchmark file is named layer_1_256
3. For compiling any code (CPU or GPU) use the following :
	nvcc -o "executable_file_name" "filename.cu"
   example: nvcc -o sample_GPU_code sample_GPU_code.cu
4. For running the executable of GPU code do the following:
	nvprof ./"executable_file_name" "batch_size"
   example: nvprof ./sample_GPU_code 32; where 32 is batch size
   For running the executable of CPU code do the following:
	./"executable_file_name" "batch_size"
   example: ./sample_CPU_code 32; where 32 is batch size
5. PLEASE NOTE: Before running a code which has multi-kernel implementation (file name has "mk" in it) especially for larger batches,
PLEASE MAKE SURE THAT NO OTHER PROCESS IS RUNNING ON GPU because multi-kernel implementation for larger batches allocates a lot of gloabl memory.
6. There is a set of files which I did just for experiment (file name has "experiment" in it). It is a less precise version of my best case implementation but it does not allocates huge global memory.
This lower precision of this implementation effects smaller batches like 32,1 but for higher batches, this scales well. 
7. The GPU code just prints the max_error value that it calculates after comparing the GPU result with corresponding benchmark.
8. For any query send an email to rnandyma@purdue.edu 