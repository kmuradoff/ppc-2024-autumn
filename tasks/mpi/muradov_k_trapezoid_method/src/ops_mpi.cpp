#include "mpi/muradov_k_trapezoidal_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <vector>

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralSequential::pre_processing() {
    a_ = *reinterpret_cast<double*>(taskData->inputs[0]);
    b_ = *reinterpret_cast<double*>(taskData->inputs[1]);
    n_ = *reinterpret_cast<int*>(taskData->inputs[2]);
    return true;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralSequential::validation() {
    return taskData->outputs_count[0] == 1;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralSequential::run() {
    res_ = integrate_function(a_, b_, n_, function_);
    return true;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralSequential::post_processing() {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res_;
    return true;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel::pre_processing() {
    if (world.rank() == 0) {
        a_ = *reinterpret_cast<double*>(taskData->inputs[0]);
        b_ = *reinterpret_cast<double*>(taskData->inputs[1]);
        n_ = *reinterpret_cast<int*>(taskData->inputs[2]);
    }
    return true;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel::validation() {
    if (world.rank() == 0) {
        return taskData->outputs_count[0] == 1;
    }
    return true;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel::run() {
    double params[3] = {0.0};
    if (world.rank() == 0) {
        params[0] = a_;
        params[1] = b_;
        params[2] = static_cast<double>(n_);
    }
    boost::mpi::broadcast(world, params, std::size(params), 0);
    double local_res = integrate_function(params[0], params[1], static_cast<int>(params[2]), function_);
    boost::mpi::reduce(world, local_res, res_, std::plus(), 0);
    return true;
}

bool muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel::post_processing() {
    if (world.rank() == 0) {
        *reinterpret_cast<double*>(taskData->outputs[0]) = res_;
    }
    return true;
}

void muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralSequential::set_function(
    const std::function<double(double)>& f) {
    function_ = f;
}

void muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel::set_function(
    const std::function<double(double)>& f) {
    function_ = f;
}

double muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralSequential::integrate_function(
    double a, double b, int n, const std::function<double(double)>& f) {
    const double width = (b - a) / n;
    double result = 0.0;
    for (int step = 0; step < n; step++) {
        const double x1 = a + step * width;
        const double x2 = a + (step + 1) * width;
        result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
    }
    return result;
}

double muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel::integrate_function(
    double a, double b, int n, const std::function<double(double)>& f) {
    int rank = world.rank();
    int size = world.size();

    const double width = (b - a) / n;
    double result = 0.0;
    for (int step = rank; step < n; step += size) {
        const double x1 = a + step * width;
        const double x2 = a + (step + 1) * width;
        result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
    }
    return result;
}
