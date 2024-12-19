// Copyright 2024 Muradov Kamal
#include <gtest/gtest.h>
#include "mpi/muradov_k_trapezoid_method/include/ops_mpi.hpp"
#include <cmath>

double test_function(double x) {
    return x * x; // Example function for integration (f(x) = x^2)
}

TEST(MPI_Trapezoid, IntegrationTest) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        double result = trapezoidal_method(test_function, 0.0, 1.0, 100);
        ASSERT_NEAR(result, 1.0 / 3.0, 1e-6); // Known integral of x^2 from 0 to 1
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}