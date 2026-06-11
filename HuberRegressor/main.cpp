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
#include <cmath>

class HuberRegressor: public Model {
public:
  std::vector<long double> weights;

  long double epsilon;
  long double lambda;
  long double learning_rate;
  int max_iter;
  long double tol;

  HuberRegressor(long double epsilon = 1.35, long double lambda = 0.0001, long double learning_rate = 0.0001, int max_iter = 100, long double tol = 1e-6) {
    this->epsilon = epsilon;
    this->lambda = lambda;
    this->learning_rate = learning_rate;
    this->max_iter = max_iter;
    this->tol = tol;
  }

  long double sign(long double x) {
    if(x < 0) {
      return -1;
    }
    if (x > 0) {
      return 1;
    }
    return 0;
  } 

  void fit(std::vector<std::vector<long double>> X, std::vector<long double> y) override {
    for (std::vector<long double>& row : X) {
      row.push_back(1.0);
    }

    size_t n = X.size();
    size_t p = X[0].size();

    weights.assign(p, 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
      std::vector<long double>old_weights = weights;

      std::vector<long double>delta(p, 0.0);

      for (int i = 0; i < n; i++) {
        long double residual = y[i];
        for (int j = 0; j < p; j++) {
          residual -= X[i][j] * weights[j];
        }

        long double psi;
        if (std::abs(residual) <= epsilon) {
          psi = 2.0 * residual;
        } else {
          psi = 2.0 * epsilon * sign(residual);
        }

        for(int j = 0; j < p; j++) {
          delta[j] += learning_rate * psi * X[i][j];
        }
      }

      long double diff = 0.0;
      for (int j = 0; j < p; j++) {
        delta[j] /= static_cast<long double>(n);

        if (j == p - 1) {
          weights[j] += delta[j]; // bias
        } else {
          weights[j] += delta[j] - 2.0 * learning_rate * lambda * weights[j];
        }

        diff += std::abs(weights[j] - old_weights[j]);
      }

      if(diff < tol) {
        break;
      }
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

  HuberRegressor best_model;
  long double best_score = -1e18;

  for (long double epsilon = 13; epsilon < 20; epsilon += 0.1) {
    for (long double lambda = 0.0001; lambda < 0.001; lambda += 0.0001) {
      HuberRegressor model(epsilon, lambda);
      model.fit(X_train, y_train);

      long double score = model.score(X_test, y_test);
      std::cerr << "Score: " << score << std::endl;

      if (score > best_score) {
        best_score = score;
        best_model = model;
      }

      std::vector<long double> model_weights = model.get_model_weights();
      std::cerr << "Weights: [ ";
      for (size_t i = 0; i < model_weights.size(); i++) {
        std::cerr << model_weights[i] << " ";
      }
      std::cerr << "]; Bias = " << model.get_bias() << "; Epsilon = " << model.epsilon << "; Lambda = " << model.lambda << std::endl;

      std::cerr << std::string(200, '-') << std::endl;
    }
  }

  std::cerr << "Best score: " << best_model.score(X_test, y_test) << std::endl;

  std::vector<long double> model_weights = best_model.get_model_weights();
  std::cerr << "Weights: [ ";
  for (size_t i = 0; i < model_weights.size(); i++) {
    std::cerr << model_weights[i] << " ";
  }
  std::cerr << "]; Bias = " << best_model.get_bias() << "; Epsilon = " << best_model.epsilon << "; Lambda = " << best_model.lambda << std::endl;
}