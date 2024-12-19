// Copyright 2024 Muradov Kamal
#include "mpi/muradov_k_trapezoid_method/include/ops_mpi.hpp"
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>

double trapezoidal_method(double (*func)(double), double a, double b, int n) {
    boost::mpi::communicator world;
    double h = (b - a) / n;
    double local_a = a + world.rank() * h * (n / world.size());
    double local_b = local_a + h * (n / world.size());
    double local_integral = 0.0;

    for (int i = 0; i < n / world.size(); ++i) {
        local_integral += (func(local_a) + func(local_a + h)) * h / 2;
        local_a += h;
    }

    double total_integral;
    reduce(world, local_integral, total_integral, std::plus<double>(), 0);

    return total_integral;
}
