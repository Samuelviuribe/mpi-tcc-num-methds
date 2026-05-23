#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = 1000000;

    double a = 0.0;
    double b = 1.0;

    double h = (b - a) / n;

    long long local_n = n / size;

    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;

    double local_sum = 0.0;

    local_sum = (f(local_a) + f(local_b)) / 2.0;

    for(long long i = 1; i < local_n; i++) {

        double x = local_a + i * h;

        local_sum += f(x);
    }

    double local_integral = local_sum * h;

    double total_integral = 0.0;

    MPI_Reduce(
        &local_integral,
        &total_integral,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if(rank == 0) {

        printf("Aproximacion de PI:\n");
        printf("%.15f\n", total_integral);

        printf("\nValor real:\n");
        printf("3.141592653589793\n");

        printf("\nError:\n");
        printf("%.15f\n",
               fabs(total_integral - 3.141592653589793));
    }

    MPI_Finalize();

    return 0;
}