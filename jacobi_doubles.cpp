#include "jacobi.h"

using namespace std;

/**
 * @author: Sathya Sudarshan
 *
 * @remarks: This program solves a system of equations using Jacobi's method in parallel by
 *           leveraging a grid topology.
 */

/**
 * Creates a 2D grid communicator
 */
MPI_Comm create_grid_comm(int p, MPI_Comm comm)
{
    int q = sqrt(p);

    /*
       We assume that the number of processors is a perfect square: p = q ×q with q = √p,
       arranged into a mesh of size q ×q.
    */
    if (q * q != p)
    {
        cerr << "Number of processors must be a perfect square" << endl;
        exit(1);
    }

    vector<int> dims(2);
    dims[0] = dims[1] = q;
    vector<int> periods(2);
    periods[0] = periods[1] = 0;
    MPI_Comm grid;
    MPI_Cart_create(comm, 2, &dims[0], &periods[0], 0, &grid);
    return grid;
}

/**
 * Compute the send counts and displacements for communication
 */
void sendcounts_and_displacements(int n, int size, int *send_counts, int *displs, int sum)
{
    int rem = n % size;

    for (int i = 0; i < size; i++)
    {
        send_counts[i] = n / size;

        if (rem-- > 0)
        {
            send_counts[i]++;
        }

        displs[i] = sum;
        sum += send_counts[i];
    }
}

/**
 * Distributes the vector b to the local processors
 */
void distribute_vec(double *b, double **local_b, MPI_Comm grid, int n)
{
    int world_rank, world_size, grid_rank, coords[2];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    *local_b = new double[local_size[0]]();

    int *sendcounts = new int[dims[0]]();
    int *displs = new int[dims[0]]();

    // Create the column communicator
    MPI_Comm column_comm;
    int remain_dims[2] = {1, 0};
    MPI_Cart_sub(grid, remain_dims, &column_comm);

    if (!coords[1])
    {
        sendcounts_and_displacements(n, dims[0], sendcounts, displs, 0);
        MPI_Scatterv(b, sendcounts, displs, MPI_DOUBLE, *local_b, local_size[0], MPI_DOUBLE, 0, column_comm);
    }

    MPI_Comm_free(&column_comm);
    MPI_Barrier(grid);
}

/**
 * Distributes the matrix A to the local processors
 */
void distribute_matrix(double *A, double **local_A, MPI_Comm grid, int n)
{
    int world_rank, world_size, grid_rank, coords[2];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    double *column_distributed = new double[local_size[0] * n]();

    // Create the column communicator
    MPI_Comm column_comm;
    int remain_dims[2] = {1, 0};
    MPI_Cart_sub(grid, remain_dims, &column_comm);

    if (!coords[1])
    {
        int *sendcounts = new int[dims[0]]();
        int *displs = new int[dims[0]]();

        sendcounts_and_displacements(n, dims[0], sendcounts, displs, 0);

        // Sendcounts are multiplied by n.
        for (int i = 0; i < dims[0]; i++)
        {
            sendcounts[i] *= n;
        }

        displs[0] = 0;
        for (int i = 1; i < dims[0]; i++)
        {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }

        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, column_distributed, n * local_size[0], MPI_DOUBLE, 0, column_comm);
    }

    MPI_Comm_free(&column_comm);
    MPI_Barrier(grid);

    // Create a row communicator
    MPI_Comm row_comm;

    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(grid, remain_dims, &row_comm);

    (*local_A) = new double[local_size[0] * local_size[1]]();

    MPI_Barrier(grid);

    int *sendcounts = new int[dims[1]]();
    int *displs = new int[dims[1]]();

    sendcounts_and_displacements(n, dims[1], sendcounts, displs, 0);

    // Get the rank of the root for the row communicator this proc is in.
    int row_comm_root;
    int row_root_coords[] = {0};
    MPI_Cart_rank(row_comm, row_root_coords, &row_comm_root);

    for (int i = 0; i < local_size[0]; i++)
    {
        // //Perform the scatter
        MPI_Scatterv((column_distributed + i * n), sendcounts, displs, MPI_DOUBLE, (*local_A + i * local_size[1]), local_size[1], MPI_DOUBLE, row_comm_root, row_comm);
    }

    MPI_Comm_free(&row_comm);
    MPI_Barrier(grid);
}

/**
 * Transpose a vector by sending from (i, 0) to (i, i), and then bcast along the column.
 */
void transpose_vector(double *x, double *x_t, MPI_Comm grid, int n)
{
    int world_rank, world_size, grid_rank, coords[2];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    // Create the column communicator
    MPI_Comm column_comm;
    int remain_dims[2] = {1, 0};
    MPI_Cart_sub(grid, remain_dims, &column_comm);

    // Create the row communicator
    MPI_Comm row_comm;
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(grid, remain_dims, &row_comm);

    if (coords[0] == 0 && coords[1] == 0)
    {
        MPI_Sendrecv(x, local_size[0], MPI_DOUBLE,
                     coords[0], 111,
                     x_t, local_size[0], MPI_DOUBLE,
                     0, 111, row_comm, MPI_STATUS_IGNORE);
    }
    else
    {
        if (coords[1] == 0)
        {
            MPI_Send(x, local_size[0], MPI_DOUBLE, coords[0], 111, row_comm);
        }
        else if (coords[1] == coords[0])
        {
            MPI_Recv(x_t, local_size[0], MPI_DOUBLE, 0, 111, row_comm, MPI_STATUS_IGNORE);
        }
    }

    // Bcast from (i, i) Along the column
    MPI_Bcast(x_t, local_size[1], MPI_DOUBLE, coords[1], column_comm);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);

    MPI_Barrier(grid);
}

/**
 * Perform the matrix vector multiplication
 */
void matvecmult(double *local_A, double *local_x, double *local_y, MPI_Comm grid, int n)
{
    int world_rank, world_size, grid_rank, coords[2];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    // Create a row communicator
    MPI_Comm row_comm;
    int remain_dims[2] = {0, 1};
    MPI_Cart_sub(grid, remain_dims, &row_comm);

    // Create a column communicator
    MPI_Comm column_comm;
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid, remain_dims, &column_comm);

    // Find y = Ax locally

    double *local_x_t = new double[local_size[1]];
    transpose_vector(local_x, local_x_t, grid, n);

    double *smaller_y = new double[local_size[0]];

    for (int i = 0; i < local_size[0]; i++)
    {
        double sum = 0;
        for (int j = 0; j < local_size[1]; j++)
        {

            sum += local_A[i * local_size[1] + j] * local_x_t[j];
        }
        smaller_y[i] = sum;
    }

    // Reduce the smaller_y to local_y
    MPI_Reduce(smaller_y, local_y, local_size[0], MPI_DOUBLE, MPI_SUM, 0, row_comm);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
}

/**
 * Compute d and R from A
 */
void compute_d_R(double *A, double **d, double **R, int n)
{
    for (int i = 0; i < n; i++)
    {
        (*d)[i] = A[i * n + i];
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                (*R)[i * n + j] = A[i * n + j];
            }
        }
    }
}

/**
 * Perform the Jacobi iteration
 */
void jacobi(double *local_A, double *local_x, double *local_b, double *local_d, double *local_R, MPI_Comm grid, int n)
{
    int world_rank, world_size, grid_rank, coords[2];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    // Create a row communicator
    MPI_Comm row_comm;
    int remain_dims[2] = {0, 1};
    MPI_Cart_sub(grid, remain_dims, &row_comm);

    // Create a column communicator
    MPI_Comm column_comm;
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid, remain_dims, &column_comm);

    double l2_total;
    int br_flag = 0;

    long double thres = 1e-9;

    for (long int i = 0; i <= 100000000; i++)
    {

        // Compute local y
        double *local_y = new double[local_size[0]]();

        matvecmult(local_A, local_x, local_y, grid, n);
        if (!coords[1])
        {
            double l2 = 0;
            for (int j = 0; j < local_size[0]; j++)
            {
                l2 += pow(local_b[j] - local_y[j], 2);
            }

            MPI_Allreduce(&l2, &l2_total, 1, MPI_DOUBLE, MPI_SUM, column_comm);
            l2_total = sqrt(l2_total);
        }

        // Determine termination condition
        if (!coords[1])
        {
            if (l2_total < thres)
            {
                br_flag = 1;
            }
        }

        // Get the rank of the root for the row communicator this proc is in.
        int row_comm_root;
        int row_root_coords[] = {0};
        MPI_Cart_rank(row_comm, row_root_coords, &row_comm_root);

        MPI_Bcast(&br_flag, 1, MPI_INT, row_comm_root, row_comm);

        if (br_flag)
        {
            break;
        }

        matvecmult(local_R, local_x, local_y, grid, n);

        // Compute local x as x = (b - y) / d on the first column
        if (!coords[1])
        {
            for (int j = 0; j < local_size[0]; j++)
            {
                local_x[j] = (local_b[j] - local_y[j]) / local_d[j];
            }
        }
    }
}

/**
 * Gather the results from all the processors
 */
void gather_results(double *x, double *local_x, MPI_Comm grid, int n)
{
    int world_rank, world_size, grid_rank, coords[2];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    int remain_dims[2] = {0, 0};
    // Create a column communicator
    MPI_Comm column_comm;
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid, remain_dims, &column_comm);

    if (!coords[1])
    {
        int *sendcounts = new int[dims[0]]();
        int *displs = new int[dims[0]]();
        sendcounts_and_displacements(n, dims[0], sendcounts, displs, 0);
        MPI_Gatherv(local_x, local_size[0], MPI_DOUBLE, x, sendcounts, displs, MPI_DOUBLE, 0, column_comm);
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    int world_rank, world_size, grid_rank, coords[2];

    MPI_Comm grid;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    grid = create_grid_comm(world_size, MPI_COMM_WORLD);

    MPI_Comm_rank(grid, &grid_rank);
    MPI_Cart_coords(grid, grid_rank, 2, coords);

    int dims[2] = {(int)sqrt(world_size), (int)sqrt(world_size)};
    int n;

    string matrix_input_file = argv[1];
    string vector_input_file = argv[2];
    string output_file = argv[3];

    if (!world_rank)
    {
        ifstream file(matrix_input_file);
        file >> n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size[2] = {0, 0};
    local_size[0] = n / dims[0] + (coords[0] < n % dims[0] ? 1 : 0);
    local_size[1] = n / dims[1] + (coords[1] < n % dims[1] ? 1 : 0);

    double *A = new double[n * n]();
    double *b = new double[n]();
    double *x = new double[n]();

    // Get matrix A and vector b input
    if (!world_rank)
    {
        ifstream matrix_file(matrix_input_file);
        matrix_file >> n;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix_file >> A[i * n + j];
            }
        }

        ifstream vector_file(vector_input_file);
        for (int i = 0; i < n; i++)
        {
            vector_file >> b[i];
        }
    }

    // Initialize the local A, b, and x.
    double *local_A = NULL;
    double *local_b = NULL;
    double *local_x = new double[local_size[0]]();

    distribute_vec(b, &local_b, grid, n);

    distribute_matrix(A, &local_A, grid, n);

    double *local_d = NULL;
    double *local_R = NULL;

    double *d = new double[n]();
    double *R = new double[n * n]();

    compute_d_R(A, &d, &R, n);

    distribute_vec(d, &local_d, grid, n);
    distribute_matrix(R, &local_R, grid, n);

    double start_time = MPI_Wtime();
    jacobi(local_A, local_x, local_b, local_d, local_R, grid, n);

    // Gather the local x
    gather_results(x, local_x, grid, n);

    double end_time = MPI_Wtime();

    // Write the results to a file
    if (!world_rank)
    {
        ofstream file(output_file);
        for (int i = 0; i < n; i++)
        {
            file << setprecision(19) << x[i] << " ";
        }
    }

    if (!world_rank)
    {
        cout << "Time: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
