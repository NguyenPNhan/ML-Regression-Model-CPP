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
#include <string>

class Ridge: public Model {
public:
  std::vector<long double> weights;
  long double alpha;

  Ridge(long double alpha = 1) {
    this->alpha = alpha;
  }

  void fit(std::vector<std::vector<long double>> X, std::vector<long double> y) override {
    for (std::vector<long double>& row : X) {
      row.push_back(1.0);
    }

    Matlib::Matrix mX(X);
    Matlib::Matrix mXT = Matlib::transpose(mX);
    Matlib::Matrix mXTX = Matlib::matrix_multiply(mXT, mX);

    for (int i = 0; i < mXTX.rows(); i++) {
      mXTX.value[i][i] += static_cast<long double>(X.size()) * alpha;
    }

    Matlib::Matrix mXTX_inverse = Matlib::inverse(mXTX);
    
    Matlib::Matrix mXTy = Matlib::matrix_multiply(mXT, Matlib::Matrix(y));
    Matlib::Matrix mW = Matlib::matrix_multiply(mXTX_inverse, mXTy);
    
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

  std::vector<long double>get_model_weights() {
    return weights;
  }
};

int main() {
  std::cerr << std::fixed << std::setprecision(4);

  Pandas::DataFrame df = Pandas::read_csv("../data.csv");

  Pandas::DataFrame df_train, df_test;
  std::tie(df_train, df_test) = train_test_split(df);

  std::vector<std::vector<long double>> X_train = df_train.drop({"price"}).data, X_test = df_test.drop({"price"}).data;
  std::vector<long double> y_train = df_train.get_column_data("price"), y_test = df_test.get_column_data("price");

  Ridge best_model;
  long double best_score = -1e18;

  for (long double alpha = 0; alpha < 0.5; alpha += 0.005) {
    Ridge model(alpha);
    model.fit(X_train, y_train);

    long double score = model.score(X_test, y_test);
    std::cerr << "Score: " << score << std::endl;

    if (score > best_score) {
      best_score = score;
      best_model = model;
    }

    std::vector<long double>model_weights = model.get_model_weights();
    std::cerr << "Weights: [ ";
    for (size_t i = 0; i + 1 < model_weights.size(); i++) {
      std::cerr << model_weights[i] << " ";
    }
    std::cerr << "]; Bias = " << model_weights.back() << "; Alpha = " << model.alpha << std::endl;

    std::cerr << std::string(200, '-') << std::endl;
  }

  std::cerr << "Best score: " << best_model.score(X_test, y_test) << std::endl;

  std::vector<long double>model_weights = best_model.get_model_weights();
  std::cerr << "Weights: [ ";
  for (size_t i = 0; i + 1 < model_weights.size(); i++) {
    std::cerr << model_weights[i] << " ";
  }
  std::cerr << "]; Bias = " << model_weights.back() << "; Alpha = " << best_model.alpha << std::endl;
}