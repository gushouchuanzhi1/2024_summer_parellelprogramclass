Homework 7
– Task 1: Implement the parallel algorithm for Gaussian 
elimination with partial pivoting using 1D column block cyclic 
partitioning for data distribution
– Task 2: Design an efficient parallel algorithm for Gaussian 
elimination with partial pivoting for matrix with a special form. 
In this matrix all the elements in the upper triangle are zeros 
except 𝑐𝑐𝑖𝑖,𝑖𝑖+1, e.g., assume n = 7
𝑐𝑐0,0 𝑐𝑐0,1
𝑐𝑐1,0 𝑐𝑐1,1 𝑐𝑐1,2
𝑐𝑐2,0 𝑐𝑐2,1 𝑐𝑐2,2 𝑐𝑐2,3
𝑐𝑐3,0 𝑐𝑐3,1 𝑐𝑐3,2 𝑐𝑐3,3 𝑐𝑐3,4
𝑐𝑐4,0 𝑐𝑐4,1 𝑐𝑐4,2 𝑐𝑐4,3 𝑐𝑐4,4 𝑐𝑐4,5
𝑐𝑐5,0 𝑐𝑐5,1 𝑐𝑐5,2 𝑐𝑐5,3 𝑐𝑐5,4 𝑐𝑐5,5 𝑐𝑐5,6
𝑐𝑐6,0 𝑐𝑐6,1 𝑐𝑐6,2 𝑐𝑐6,3 𝑐𝑐6,4 𝑐𝑐6,5 𝑐𝑐6,6
