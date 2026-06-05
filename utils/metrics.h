#pragma once

#include <vector>

long double mean_absolute_error(const std::vector<long double> y_true, const std::vector<long double> y_pred);
long double mean_squared_error(const std::vector<long double> y_true, const std::vector<long double> y_pred);
long double r2_score(const std::vector<long double> y_true, const std::vector<long double> y_pred);
