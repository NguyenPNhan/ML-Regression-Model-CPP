#include "metrics.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>

long double mean_absolute_error(const std::vector<long double> y_true, const std::vector<long double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::invalid_argument("mean_absolute_error: y_true and y_pred must have the same size");
  }
  if (y_true.empty()) {
    throw std::invalid_argument("mean_absolute_error: input vector must not be empty");
  }

  long double sum = 0;
  for (size_t i = 0; i < y_true.size(); i++) {
    long double diff = y_true[i] - y_pred[i];
    sum += diff * diff;
  }
  return sum / static_cast<long double>(y_true.size());
}

long double mean_squared_error(const std::vector<long double> y_true, const std::vector<long double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::invalid_argument("mean_absolute_error: y_true and y_pred must have the same size");
  }
  if (y_true.empty()) {
    throw std::invalid_argument("mean_absolute_error: input vector must not be empty");
  }

  long double sum = 0;
  for (size_t i = 0; i < y_true.size(); i++) {
    long double diff = y_true[i] - y_pred[i];
    sum += diff * diff;
  }
  return sum / static_cast<long double>(y_true.size());
}

#include <iostream>

long double r2_score(const std::vector<long double> y_true, const std::vector<long double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    throw std::invalid_argument("r2_score: y_true and y_pred must have the same size");
  }
  if (y_true.empty()) {
    throw std::invalid_argument("r2_score: input vector must not be empty");
  }

  long double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0L) / static_cast<long double>(y_true.size());


  long double ss_res = 0, ss_tot = 1e-9;
  for (size_t i = 0; i < y_true.size(); i++) {
    long double residual = y_true[i] - y_pred[i];
    ss_res += residual * residual;

    long double diff = y_true[i] - mean;
    ss_tot += diff * diff;
  }

  return 1.0L - ss_res / ss_tot;
}