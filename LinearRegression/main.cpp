#include "../utils/pandas.h"
#include "../utils/model_selection.h"
#include "../utils/matlib.h"
#include "../utils/metrics.h"
#include "../class/model.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <tuple>

class LinearRegression: public Model {
  void fit(std::vector<std::vector<long double>> X, std::vector<long double> y) override {
    
  }
  std::vector<long double> predict(std::vector<std::vector<long double>> X) override {
    return {};
  }
};

int main() {
  Pandas::DataFrame df = Pandas::read_csv("../data.csv");

  Pandas::DataFrame df_train, df_test;
  std::tie(df_train, df_test) = train_test_split(df);

  
}