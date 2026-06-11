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

class Lasso: public Model {
public:
  std::vector<long double> weights;

  long double lambda;
  int max_iter;
  long double tol;

  Lasso(long double lambda = 0.1, int max_iter = 1000, long double tol = 1e-6) {
    this->lambda = lambda;
    this->max_iter = max_iter;
    this->tol = tol;
  }

  long double soft_threshold(long double rho, long double alpha) {
    if (rho > alpha) {
      return rho - alpha;
    }
    if (rho < -alpha) {
      return rho + alpha;
    }
    return 0.0;
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

      for (size_t j = 0; j < p; j++) {
        long double rho = 0, z = 0;

        for (size_t i = 0; i < n; i++) {
          long double residual = y[i];
          for (size_t k = 0; k < p; k++) {
            if (k != j) {
              residual -= X[i][k] * weights[k];
            } 
          }

          rho += X[i][j] * residual;
          z += X[i][j] * X[i][j];
        }
  

        if (j + 1 < p) {
          weights[j] = rho / z; //bias
        } else {
          weights[j] = soft_threshold(rho, lambda / 2.0) / z;
        }
      }

      long double diff = 0;
      for (size_t j = 0; j < p; j++) {
        diff += std::abs(weights[j] - old_weights[j]);
      }

      if (diff < tol) {
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

  Lasso best_model;
  long double best_score = -1e18;

  for (long double lambda = 2e4; lambda < 3e4; lambda += 1e2) {
    Lasso model(lambda);
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
    std::cerr << "]; Bias = " << model_weights.back() << "; Lambda = " << model.lambda << std::endl;

    std::cerr << std::string(200, '-') << std::endl;
  }

  std::cerr << "Best score: " << best_model.score(X_test, y_test) << std::endl;

  std::vector<long double>model_weights = best_model.get_model_weights();
  std::cerr << "Weights: [ ";
  for (size_t i = 0; i < model_weights.size(); i++) {
    std::cerr << model_weights[i] << " ";
  }
  std::cerr << "]; Bias = " << best_model.get_bias() << "; Lambda = " << best_model.lambda << std::endl;
}