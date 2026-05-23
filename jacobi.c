#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 3
#define ITERACIONES 20
#define TOL 0.000001

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != N) {

        if(rank == 0) {
            printf("Este programa necesita exactamente 3 procesos.\n");
        }

        MPI_Finalize();
        return 0;
    }

    double A[N][N] = {
        {10, -1, 2},
        {-1, 11, -1},
        {2, -1, 10}
    };

    double b[N] = {6, 25, -11};

    double x[N] = {0, 0, 0};

    double x_new[N] = {0, 0, 0};

    double suma;
    double error;

    for(int iter = 0; iter < ITERACIONES; iter++) {

        suma = 0.0;

        for(int j = 0; j < N; j++) {

            if(j != rank) {
                suma += A[rank][j] * x[j];
            }
        }

        x_new[rank] =
            (b[rank] - suma) / A[rank][rank];

        MPI_Allgather(
            &x_new[rank],
            1,
            MPI_DOUBLE,
            x_new,
            1,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        error = fabs(x_new[rank] - x[rank]);

        for(int i = 0; i < N; i++) {
            x[i] = x_new[i];
        }

        if(rank == 0) {

            printf("Iteracion %d\n", iter + 1);

            printf("x = %.6f\n", x[0]);
            printf("y = %.6f\n", x[1]);
            printf("z = %.6f\n", x[2]);

            printf("--------------------------\n");
        }

        if(error < TOL) {
            break;
        }
    }

    if(rank == 0) {

        printf("\nSolucion final aproximada:\n");

        printf("x = %.6f\n", x[0]);
        printf("y = %.6f\n", x[1]);
        printf("z = %.6f\n", x[2]);
    }

    MPI_Finalize();

    return 0;
}