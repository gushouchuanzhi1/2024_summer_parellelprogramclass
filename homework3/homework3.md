Homework 3: Parallel Scan Operation
– Implement the parallel algorithm for scan operation using 
OpenMP
– Compare the performance with the sequential algorithm
– In a parallel region, if we want part of code to be done by a 
single (any) thread, we can use single directive
#pragma omp single
– All other threads will skip the single region and stop at the 
barrier at the end of the single construct until all threads have 
reached the barrier


我们需要实现顺序扫描的函数，并且需要比较顺序执行和并行执行的效率。