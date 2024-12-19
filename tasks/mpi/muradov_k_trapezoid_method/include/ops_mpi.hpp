// Copyright 2024 Muradov Kamal
#pragma once

#include <boost/mpi/communicator.hpp>
#include <vector>

double trapezoidal_method(double (*func)(double), double a, double b, int n);