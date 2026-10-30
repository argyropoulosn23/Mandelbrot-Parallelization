#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

extern void matToImage(char* filename, int* mat, int* dims);
extern void matToImageColor(char* filename, int* mat, int* dims);

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double programStart = MPI_Wtime();
    int nx=6000;
    int ny=4000;
    double xStart=-2;
    double xEnd=1;
    double yStart=-1;
    double yEnd=1;
    int maxIter=255;

    //Master rank stores entire image matrix
    int* matrix = NULL;
    if(rank == 0){
        matrix=(int*)malloc(nx*ny*sizeof(int));

        if(matrix == NULL){
            printf("Could not allocate matrix\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //Gives each worker memory to store a row
    int* row = (int*)malloc(nx*sizeof(int));
    if(row == NULL){
        printf("Rank %d: Could not allocate row\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //------------------Master------------------
    if(rank == 0){
        if(size == 1){

            //This is just for the situation where theres only one process
            for(int i=0;i<ny;i++){//rows
                for(int j=0;j<nx;j++){//col
                    //c=x0+iy0
                    double x0=xStart+(1.0*j/nx)*(xEnd-xStart);
                    double y0=yStart+(1.0*i/ny)*(yEnd-yStart);
                    //z=0=x+iy
                    double x=0;
                    double y=0;

                    int iter=0;
                    //do the equation (z=z^2+c)
                    while(iter<maxIter){
                        double temp=x*x-y*y+x0;
                        y=2*x*y+y0;
                        x=temp;
                        if(x*x+y*y>4){
                            break;
                        }
                        iter++;
                    }
                    matrix[i*nx+j]=iter;
                }
            }
        } else {
            int nextRow = 0;
            int completedRows = 0;

            for(int worker=1; worker<size; worker++){

                if(nextRow < ny){

                    MPI_Send(&nextRow, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);

                    nextRow++;
                }
                else{

                    int stop = -1;

                    MPI_Send(&stop, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
                }
            }

            while(completedRows < ny){
                MPI_Status status;
                int completedRow;

                MPI_Recv(&completedRow, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);

                int worker = status.MPI_SOURCE;

                MPI_Recv(&matrix[completedRow * nx], nx, MPI_INT, worker, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                completedRows++;

                if(nextRow < ny){
                    MPI_Send(&nextRow, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
                    nextRow++;
                }
                else{
                    int stop = -1;
                    MPI_Send(&stop, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
                }
            }
        }

        int dims[2];

        dims[0]=ny;
        dims[1]=nx;

        double programEnd = MPI_Wtime();

        printf("Total program calculation time: %f seconds\n", programEnd - programStart);

        matToImageColor("mandelbrot.jpg", matrix, dims);

        free(matrix);

    }else{
    //------------------Workers------------------
        int numThreads = omp_get_max_threads();

        double* threadTimes = (double*)calloc(numThreads, sizeof(double));

        if(threadTimes == NULL){
            printf("Rank %d: Could not allocate thread timing array\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double rankCalculationTime = 0.0;

        while(1){

            int i;

            MPI_Recv(&i, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if(i == -1){
                break;
            }

            double start = MPI_Wtime();

            #pragma omp parallel
            {
                int threadID = omp_get_thread_num();

                double threadStart = omp_get_wtime();

                #pragma omp for
                for(int j=0;j<nx;j++){

                    double x0=xStart+(1.0*j/nx)*(xEnd-xStart);
                    double y0=yStart+(1.0*i/ny)*(yEnd-yStart);

                    double x=0;
                    double y=0;

                    int iter=0;

                    while(iter<maxIter){

                        double temp=x*x-y*y+x0;
                        y=2*x*y+y0;
                        x=temp;

                        if(x*x+y*y>4){
                            break;
                        }

                        iter++;
                    }

                    row[j]=iter;
                }

                double threadEnd = omp_get_wtime();

                threadTimes[threadID] += threadEnd - threadStart;
            }

            double end = MPI_Wtime();
            rankCalculationTime += end - start;
            MPI_Send(&i, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

            MPI_Send(row, nx, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }

        printf("Rank %d calculation time: %f seconds\n", rank, rankCalculationTime);

        if(rank == 1){
            for(int t=0; t<numThreads && t<10; t++){
                printf("Rank %d Thread %d calculation time: %f seconds\n", rank, t, threadTimes[t]);
            }
        }
        free(threadTimes);
    }

    free(row);

    MPI_Finalize();

    return 0;
}

