#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>


//calculates the average of the adjacent elements for each element in a given column
//"left" is the column to the immediate left of the sub array. similar for "right"
//"n" is the number of rows, "m" is the number of columns
int ave(double** read, double** write, double* left, double* right, int n, int c, double prec, int rank, int size, int min, int max) {
    int succ = 1;
    int row, col;

    //if max index is greater than the size of the whole nxn array reduce c, the number of columns to be worked on by this processor
    if (max > n - 1)
        c = c - (max - n);
   
    //top and bottom row remain constant so ignore these rows
    for (row = 1; row < n - 1; row++) {
        
        //if not the first thread, then left most column will need to be averaged
        //if it is, then it should stay the same
        if (rank != 0)
            //write[row][0] = read[row][0];
        //else
            write[row][0] = (read[row + 1][0] + read[row - 1][0] + read[row][1] + left[row]) / 4;
        //check for unsuccessfulness
        if (fabs(write[row][0] - read[row][0]) > prec)
            succ = 0;

        //get average of all elements not on the edge of the local array
        for (col = 1; col < c - 1; col++) {
            write[row][col] = (read[row + 1][col] + read[row - 1][col] + read[row][col + 1] + read[row][col - 1]) / 4;
            //check for unsuccessfulness
            if (fabs(write[row][col] - read[row][col]) > prec)
                succ = 0;
        }

        //if not the last thread, then right most column will need to be averaged
        //if it is, then it should stay the same
        if (rank != size - 1)
            //write[row][c-1] = read[row][c-1];
        
        //else
            write[row][c - 1] = (read[row + 1][c - 1] + read[row - 1][c - 1] + right[row] + read[row][c - 2]) / 4;

        //check for unsuccessfulness
        if (fabs(write[row][c - 1] - read[row][c - 1]) > prec)
            succ = 0; 
    }
    return succ;
}


int main(int argc, char* argv[]) {

    int msec = 0;
    clock_t before = clock();

    int i = 0, j = 0, row = 0, col = 0;
    //size of array
    int n = 150;

    //precision required
    double prec = 0.0001;

    //rank, and number of processes
    int rc, rank, size;

    MPI_Status status;

    //processes start
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //success Bools
    int succ;
    int success;

    //number of columns each sub/local array will have
    int c = ceil((double)n / size);
    int m = c * size;

    //allocate memory for the array
    double* arr_single = calloc(m * n, sizeof(double));
    double** arr = malloc(sizeof(double) * n);

   
    for (i = 0; i < n; ++i)
        arr[i] = arr_single + i * n;

    //assign random values between 10 and 50 to the array
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            arr[i][j] = (rand() % 39) + 10.0;
        }
        for (j = n; j < m; j++) {
            arr[i][j] = 0.0;
        }
    }

    //allocate memory for the read and write arrays
    double* l_a_r_single = calloc(c * n, sizeof(double));
    double** loc_arr_read = malloc(sizeof(double) * (n));

    double* l_a_w_single = calloc(c * n, sizeof(double));
    double** loc_arr_write = malloc(sizeof(double) * (n));

    //creates the column arrays each process will send to it's neighbouring processes
    double* sendl = malloc(n * sizeof(double));
    double* sendr = malloc(n * sizeof(double));
    //space to store the neighboring columns received from other processes
    double* recvl = calloc(n, sizeof(double));
    double* recvr = calloc(n, sizeof(double));

    double** temp;

    int* succ_arr = calloc(size, sizeof(int));

    int min = rank * c;
    int max = min + c - 1;

    //creates the local read and write arrays each process will be using
    for (row = 0; row < n; row++) {
        loc_arr_read[row] = l_a_r_single + row * c;
        loc_arr_write[row] = l_a_w_single + row * c;
        for (col = 0; col < c; col++) {
            loc_arr_read[row][col] = arr[row][col + (rank * c)];
            loc_arr_write[row][col] = arr[row][col + (rank * c)];
        }
    }

    do {
        success = 1;
        succ = 1;

        //assign values to be sent to neighboring processors
        for (i = 0; i < n; i++) {
            sendl[i] = loc_arr_read[i][0];
            sendr[i] = loc_arr_read[i][c - 1];
        }

        //sends and received data from neighboring processes
        //uses Sendrecv as one method instead of separately for thread safety
        if (size > 1) {
            if (rank == 0)
                MPI_Sendrecv(sendr, n, MPI_DOUBLE, rank + 1, 1, recvr, n, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);
            else if (rank == size - 1)
                MPI_Sendrecv(sendl, n, MPI_DOUBLE, rank - 1, 1, recvl, n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
            else {
                MPI_Sendrecv(sendr, n, MPI_DOUBLE, rank + 1, 1, recvr, n, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);
                MPI_Sendrecv(sendl, n, MPI_DOUBLE, rank - 1, 1, recvl, n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
            }
        }

        //calculates and updates the averages, and returns whether it is precise enough
        succ = ave(loc_arr_read, loc_arr_write, recvl, recvr, n, c, prec, rank, size, min, max);
        //if a single process returned 0 "succ" becomes 0
        
        //rank 0 colects the successfulness of each processor 
        if (rank == 0) {
            for (i = 1; i < size; i++) {
                success *= succ;
                MPI_Recv(&succ, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            }
            success *= succ;

        }
        //each processor not rank 0 sends their successfulness
        else {
            MPI_Send(&succ, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }

        //no need for barrier here because MPI_Send and MPI_Recieve act as a barrier of sorts
        //ie there's no chance rank 0 will send the combined success Bool before it has seen all the ohter succ values

        //rank == 0 send the overall success to all other processors
        if (rank == 0) {
            for (i = 1; i < size; i++)
                MPI_Send(&success, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
        else
            //all other threads wait to recieve the success so they know whether to do another iteration
            MPI_Recv(&success, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);


        //swaps the read and write array
        temp = loc_arr_write;
        loc_arr_write = loc_arr_read;
        loc_arr_read = temp;

        //wait for all processors before continuing to ensure every thread is on the same iteration's value of "success"
        MPI_Barrier(MPI_COMM_WORLD);

    } while (!success);
    
    //not sure what went wrong here. MPi_Gather kept throwing errors no matter the arguments supplied
    //ideally rank == 0 would have collected all the local arrays and appended them together

  
    /*
    //use barrier to make sure all thread are synced
    MPI_Barrier(MPI_COMM_WORLD);
    
    double* arr_Fin_single = calloc(m * n, sizeof(double));
    double** arrFin = malloc(sizeof(double) * n);

    //assign random values between 10 and 50 to the array
    for (i = 0; i < n; ++i)
        arrFin[i] = arr_Fin_single + i * n;

    MPI_Gather(loc_arr_read, n, MPI_DOUBLE, arrFin, n * c, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("final:\n");
        for (row = 0; row < n; row++) {
            for (col = 0; col < m; col++) {
                printf("%.3f ", arr[row][col]);
            }
            printf("\n");
        }
    }
    */

    if (rank == 0) {
        clock_t difference = clock() - before;
        msec = difference * 1000 / CLOCKS_PER_SEC;

        
    }
    
    //instead of patching them together and testing a single array, I've checked each array indivually

    succ = 1;
    //checks each invidual processor's array against a sequential run
    for (row = 1; row < n - 1; row++) {
        for (col = rank * c; col < c-1; col++) {
            if(col != 0 && col != n-1)
                arr[row][rank*c+col] = (loc_arr_read[row + 1][rank * c + col] + loc_arr_read[row - 1][rank * c + col] + loc_arr_read[row][rank * c + col + 1] + loc_arr_read[row][rank * c + col - 1]) / 4;

             if (fabs(arr[row][rank * c + col] - loc_arr_read[row][rank * c + col]) > prec)
                succ = 0;
        }
    }

    printf("success from %d: %d\n",rank, succ);

    if (rank == 0) {
        printf("Time taken: %d milliseconds\n", msec);
    }

    MPI_Finalize();

    return 0;

}