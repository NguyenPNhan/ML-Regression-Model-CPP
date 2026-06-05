#pragma once

#include "../utils/metrics.h"
#include <vector>

class Model {
public:
  virtual ~Model() = default;
  
  virtual void fit(std::vector<std::vector<long double>> X, std::vector<long double> y) = 0;
  virtual std::vector<long double> predict(std::vector<std::vector<long double>> X) = 0;

  long double score(std::vector<std::vector<long double>> X, std::vector<long double> y) {
    return r2_score(y, predict(X));
  }
};