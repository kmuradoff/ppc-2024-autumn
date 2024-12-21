#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/muradov_k_trapezoidal_method/include/ops_mpi.hpp"

TEST(muradov_k_trapezoidal_method_mpi, test_pipeline_run) {
    boost::mpi::communicator world;
    double a = -2.0;
    double b = 10.0;
    int n = 100000;
    double result = 0.0;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
        taskDataPar->outputs_count.emplace_back(1);
    }

    auto testMpiTaskParallel =
        std::make_shared<muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel>(taskDataPar);
    auto f = [](double x) { return std::pow(x, 3) - std::pow(3, x) + std::exp(x); };
    testMpiTaskParallel->set_function(f);

    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
        double accurate_result = -29226.28;
        ASSERT_NEAR(accurate_result, result, 0.01);
    }
}

TEST(muradov_k_trapezoidal_method_mpi, test_task_run) {
    boost::mpi::communicator world;
    double a = -2.0;
    double b = 10.0;
    int n = 100000;
    double result = 0.0;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
        taskDataPar->outputs_count.emplace_back(1);
    }

    auto testMpiTaskParallel =
        std::make_shared<muradov_k_trapezoidal_method_mpi::TrapezoidalIntegralParallel>(taskDataPar);
    auto f = [](double x) { return std::pow(x, 3) - std::pow(3, x) + std::exp(x); };
    testMpiTaskParallel->set_function(f);

    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
        double accurate_result = -29226.28;
        ASSERT_NEAR(accurate_result, result, 0.01);
    }
}
