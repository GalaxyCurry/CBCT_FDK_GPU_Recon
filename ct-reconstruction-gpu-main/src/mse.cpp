// Small utility to compute mean squared error between two volumes.
#include "reconstruction.hpp"
#include <stdexcept>
#include <cmath>

double computeMSE(const std::vector<float>& a,
                  const std::vector<float>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "MSE error: vector size mismatch.");

    double mse = 0.0;

    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = static_cast<double>(a[i]) -
                      static_cast<double>(b[i]);
        mse += diff * diff;
    }

    return mse / static_cast<double>(a.size());
}