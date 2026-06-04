#include "metrics.h"
#include <vector>
#include <stdexcept>
#include <cmath>

long double mean_absolute_error(const std::vector<long double>y_true, const std::vector<long double>y_pred) {
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
long double mean_squared_error(const std::vector<long double>y_true, const std::vector<long double>y_pred) {
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