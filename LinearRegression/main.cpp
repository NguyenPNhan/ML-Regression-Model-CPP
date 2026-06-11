#include "../utils/pandas.h"
#include "../utils/model_selection.h"
#include "../utils/matlib.h"
#include "../utils/metrics.h"
#include "../class/model.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <tuple>
#include <iomanip>

class LinearRegression: public Model {
public:
  std::vector<long double> weights;

  void fit(std::vector<std::vector<long double>> X, std::vector<long double> y) override {
    for (std::vector<long double>& row : X) {
      row.push_back(1.0);
    }

    Matlib::Matrix mX(X);
    Matlib::Matrix mXT = Matlib::transpose(mX);
    Matlib::Matrix mXTX = Matlib::matrix_multiply(mXT, mX);
    Matlib::Matrix mXTX_inverse = Matlib::inverse(mXTX);
    
    Matlib::Matrix my(y);
    Matlib::Matrix mW = Matlib::matrix_multiply(mXTX_inverse, Matlib::matrix_multiply(mXT, my));
    
    weights.reserve(mW.value.size());
    for (std::vector<long double>& row : mW.value) {
      weights.emplace_back(row[0]);
    }
  }

  std::vector<long double> predict(std::vector<std::vector<long double>> X) override {
    for (std::vector<long double>& row : X) {
      row.push_back(1.0);
    }

    Matlib::Matrix mX(X), mW(weights);
    Matlib::Matrix my = Matlib::matrix_multiply(mX, mW);

    std::vector<long double> y;
    y.reserve(my.value.size());

    for (std::vector<long double>& row : my.value) {
      y.emplace_back(row[0]);
    } 

    return y;
  }

  std::vector<long double> get_model_weights() {
    return std::vector<long double>(weights.begin(), weights.end() - 1);
  }

  long double get_bias() {
    return weights.back();
  }
};

int main() {
  std::cerr << std::fixed << std::setprecision(4);

  Pandas::DataFrame df = Pandas::read_csv("../data.csv");

  Pandas::DataFrame df_train, df_test;
  std::tie(df_train, df_test) = train_test_split(df);

  std::vector<std::vector<long double>> X_train = df_train.drop({"price"}).data, X_test = df_test.drop({"price"}).data;
  std::vector<long double> y_train = df_train.get_column_data("price"), y_test = df_test.get_column_data("price");

  LinearRegression model;
  model.fit(X_train, y_train);

  std::cerr << "Score: " << model.score(X_test, y_test) << std::endl;

  std::vector<long double>model_weights = model.get_model_weights();
  std::cerr << "Weights: [ ";
  for (size_t i = 0; i < model_weights.size(); i++) {
    std::cerr << model_weights[i] << " ";
  }
  std::cerr << "]; Bias = " << model.get_bias() << std::endl;
}