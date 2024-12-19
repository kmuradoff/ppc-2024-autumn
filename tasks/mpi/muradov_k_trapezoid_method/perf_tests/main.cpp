// Copyright 2024 Muradov Kamal
#include <iostream>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "mpi/muradov_k_trapezoid_method/include/ops_mpi.hpp"
#include <cmath>

double perf_test_function(double x) {
    return std::sin(x);
}

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    double a = 0.0;
    double b = M_PI;
    int n = 1000000;

    double result = trapezoidal_method(perf_test_function, a, b, n);

    if (world.rank() == 0) {
        std::cout << "Integration of sin(x) from 0 to PI: " << result << "\n";
        std::cout << "Expected: 2.0" << std::endl;
    }

    return 0;
}
