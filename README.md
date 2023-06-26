# mpi-jacobi

**Parallel implementation of Jacobi's matrix method using an MPI virtual grid topology.**

## Methods

### Grid Topology:

The  designed algorithm works on a grid/mesh topology rather than a hypercubic one. Since the source topology is a mesh and our
target topology is assumed to be hypercubic, and a mapping between the two topologies is utilized. Given a perfect square number of processors $p$, the grid of processors is of dimensions $\sqrt{p}*\sqrt{p}$

### Distribution of Data:

After the embedding was set up, the matrix and vector need to be distributed. The processor with rank 0 initially holds all data. Distribute vector and distribute matrix functions calculate how much of the vector or matrix the given processor gets by taking the number of rows and floor and dividing by the dimensions of our mesh topology and if the mesh coordinate is less than the number of rows of the matrix, they add one to the local size. The vector is then block distributed along the first column. The column communicator is created using MPI_Cart_Sub, and then MPI Scatterv is used to scatter each processor’s portion from the root. For the distributed matrix, the rank 0 processor has its data along the first column, and then every processor on the first column will then scatter its data among their rows, using a row communicator which is also created using MPI_Cart_Sub. The computation of send counts and displacements takes $O(p)$ time. Communication part, Scatterv takes $O(\tau\log p + \mu n)$ to scatter along a row or a column.

### Transpose:

This section of the code was used by the matrix vector multiplication algorithm. It takes a vector that was distributed among the first column of processors and then first sends the data to the diagonal elements for the mesh. For example data that was a processor with rank (1,0) would be sent to processor with rank (1,1). After that every diagonal processor broadcasts its data within its column the transpose is successful. The sending and  receiving should take constant time, but the broadcast will take $O(\tau\log(p)+\mu m\log(p))$.

### Matrix Vector Multiplication:

After the vector gets transposed the matrix vector multiplication can be executed. The local sizes get calculated and we have two for loops compute the local matrix multiplication. After all the data gets reduced onto the first column to get the final answer. Since broadcast and reduce have essentially the same runtime, the communication in Matrix_vector Multiplication would be $O(\tau\log(\sqrt{p})+\mu(n/\sqrt{p})log(\sqrt{p})) . As for the computation part of this function, we have $O(\frac{n^2}{p})$ for finding the matrix vector products on each local processor.

### Compute d and R:

This function computes the R matrix (A - D) by removing all the diagonal elements of the original A matrix. This is done by iterating throught all the elements of A and if the elements are not diagonal elements we add those elements to R. This would have a run time of $O(n^2)$

### Jacobi’s Method Implementation:

The Jacobi function iteratively updates the solution vector x until the L2 norm of the residual vector (b-Ax) is smaller than a given threshold. The function first initializes the local row and column communicators and sets the termination threshold. Then, the function conducts the following steps iteratively: first, the function calculates the local matrix-vector multiplication and the L2 norm of the residual vector. The termination condition is obtained from the L2 norm of the residual vector, and is broadcasted to all processors using MPI_Bcast. The function then updates the local vector x based on the residual vector and the diagonal matrix. Runtime can be analyzed by looking at the functions and procedures inside. Computation wise, we have the computation part of matvecmult(), as well as the computation of y - b on the first column. With $O(n^2/p)$ , the matvecmult portion dominates this runtime. As for communication, we have AllReduce of L2, as well as the Reduce of the vector in matvecmult. So the communication runtime is dominated by $O(\tau\log(\sqrt{p})+\mu(n/\sqrt{p})\log(\sqrt{p}))$.

It is to be noted that in Jacobi, the above procedures are running repeatedly until termination condition, so that also contributed to a constant multiplier on the runtime.

## Performance Evaluation:


<img width="663" alt="image" src="https://github.com/ssudarshan7/mpi-jacobi/assets/46603681/0b82d2e8-35dc-471e-94f1-b4c402dda88a">

The above graph shows the runtime changes with 1,4,9,16,25,36 and 49 processors. The input size has a diagonally dominant matrix of size 100, and the matrix consists of random numbers. The graph shows a general trend of increasing runtime with an increasing number of processors given a fixed input. The trend can be explained by the communication overhead required for the parallel Jacobi matrix multiplication. As explained in the code design, the Jacobi algorithm requires several communication steps, including the distribution of the matrix to individual processors, MPI_Bcast, MPI_reduce, and the addition of residual vectors. Hence the runtime complexity increases with the number of processors used due to the significantly higher communication overhead between large number of processors. We observe that there is a reduction in runtime when the number of processors is equal to 25 and 36. This could be possibly explained by the fact that although there is more communication overhead with the increase in number of processors, the overall runtime is lesser because of the faster completion of the serial parts on the local matrix/vector within each processor.

<img width="592" alt="image" src="https://github.com/ssudarshan7/mpi-jacobi/assets/46603681/7f12db04-a318-4995-951b-fc4dda5d9c10">

The graph above shows the runtime change with varying input matrix size of 4, 8, 16, 32, 64, 128, 256 and 512 given a fixed processor size of 4. The graph shows a trend of an increase in runtime with higher matrix sizes. One explanation for such an increase is the communication overhead factor explained above, with the amount of data required to be distributed and exchanged among processors increasing as the input size increases. Also, the complexity increases exponentially with increase in given input is due to the fact that parallel Jacobi scales with n^2 as discussed in complexity analysis above. We also observed when the input size is small, parallel Jacobi is less efficient than serial Jacobi. This can be attributed to parallel Jacobi being dominated by the communication overhead between processors. 


