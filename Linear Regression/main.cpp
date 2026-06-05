#include "model.h"
#include "pandas.h"
#include "model_selection.h"
#include "metrics.h"
#include <vector>

class LinearRegression: public Model {
  void fit(std::vector<std::vector<long double>>X, std::vector<long double>y) override {

  }
  std::vector<long double> predict(std::vector<std::vector<long double>>X) override {
    
  }
};

int main() {

}