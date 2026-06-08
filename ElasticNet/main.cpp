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

class ElasticNet: public Model {
public:
  std::vector<long double> weights;

  long double alpha;
  long double l1_ratio;
  int max_iter;
  long double tol;

  ElasticNet(long double alpha = 0.1, long double l1_ratio = 0.5, int max_iter = 1000, long double tol = 1e-6) {
    this->alpha = alpha;
    this->l1_ratio = l1_ratio;
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

    long double lambda_1 = alpha * l1_ratio;
    long double lambda_2 = alpha * (1.0 - l1_ratio);

    weights.assign(p, 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
      std::vector<long double> old_weights = weights;

      for (size_t j = 0; j < p; j++) {
        long double rho = 0.0, z = 0;

        for (size_t i = 0; i < n; i++) {
          long double prediction_without_j = 0.0;

          for (size_t k = 0; k < p; k++) {
            if (k != j) {
              prediction_without_j += X[i][k] * weights[k];
            }
          }

          long double r_ij = y[i] - prediction_without_j;

          rho += X[i][j] * r_ij;
          z += X[i][j] * X[i][j];
        }

        if (z == 0.0) {
          weights[j] = 0.0;
          continue;
        }

        if (j == p - 1) {
          // Do not regularize bias
          weights[j] = rho / z;
        } else {
          weights[j] = soft_threshold(rho, n * lambda_1) / (z + n * lambda_2);
        }
      }

      long double diff = 0.0;
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

  ElasticNet best_model;
  long double best_score = -1e18;

  for (long double alpha = 0; alpha < 1.0; alpha += 0.01) {
    for (long double l1_ratio = 0.0; l1_ratio <= 1.0; l1_ratio += 0.1) {
      ElasticNet model(alpha, l1_ratio);
      model.fit(X_train, y_train);

      long double score = model.score(X_test, y_test);
      std::cerr << "Score: " << score << std::endl;

      if (score > best_score) {
        best_score = score;
        best_model = model;
      }

      std::vector<long double> model_weights = model.get_model_weights();
      std::cerr << "Weights: [ ";
      for (size_t i = 0; i + 1 < model_weights.size(); i++) {
        std::cerr << model_weights[i] << " ";
      }
      std::cerr << "]; Bias = " << model_weights.back() << "; Alpha = " << model.alpha << "; L1 Ratio = " << model.l1_ratio << std::endl;

      std::cerr << std::string(200, '-') << std::endl;
    }
  }

  std::cerr << "Best score: " << best_model.score(X_test, y_test) << std::endl;

  std::vector<long double> model_weights = best_model.get_model_weights();
  std::cerr << "Weights: [ ";
  for (size_t i = 0; i + 1 < model_weights.size(); i++) {
    std::cerr << model_weights[i] << " ";
  }
  std::cerr << "]; Bias = " << model_weights.back() << "; Alpha = " << best_model.alpha << "; L1 Ratio = " << best_model.l1_ratio << std::endl;
}