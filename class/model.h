#pragma once

#include <vector>

class Model {
public:
  virtual ~Model() = default;
  virtual void fit(std::vector<std::vector<long double>> X, std::vector<long double> y);
  virtual std::vector<long double> predict(std::vector<std::vector<long double>> X);
};