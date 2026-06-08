#include "../utils/pandas.h"
#include "../utils/model_selection.h"
#include "../utils/matlib.h"
#include "../utils/metrics.h"
#include "../class/model.h"
#include <vector>
#include <iostream>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <algorithm>

class BayesianRidge: public Model {
public:
  std::vector<long double> weights;

  long double alpha;
  long double lambda;
  int max_iter;
  long double tol;

  BayesianRidge(
    long double alpha = 1.0,
    long double lambda = 1.0,
    int max_iter = 300,
    long double tol = 1e-6
  ) {
    this->alpha = alpha;
    this->lambda = lambda;
    this->max_iter = max_iter;
    this->tol = tol;
  }

  void fit(std::vector<std::vector<long double>> X, std::vector<long double> y) override {
    for (std::vector<long double>& row : X) {
      row.push_back(1.0);
    }

    int n = X.size();
    int p = X[0].size();

    Matlib::Matrix mX(X);
    Matlib::Matrix mXT = Matlib::transpose(mX);
    Matlib::Matrix mXTX = Matlib::matrix_multiply(mXT, mX);
    Matlib::Matrix mXTy = Matlib::matrix_multiply(mXT, Matlib::Matrix(y));

    std::vector<long double> old_weights(p, 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
      Matlib::Matrix A = mXTX;

      for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
          A.value[i][j] *= alpha;
        }
      }

      for (int i = 0; i < A.rows() - 1; i++) {
        A.value[i][i] += lambda;
      }

      A.value[A.rows() - 1][A.cols() - 1] += 1e-12;

      Matlib::Matrix Sigma = Matlib::inverse(A);

      Matlib::Matrix b = mXTy;
      for (int i = 0; i < b.rows(); i++) {
        b.value[i][0] *= alpha;
      }

      Matlib::Matrix mW = Matlib::matrix_multiply(Sigma, b);

      weights.clear();
      weights.reserve(mW.value.size());

      for (std::vector<long double>& row : mW.value) {
        weights.emplace_back(row[0]);
      }

      std::vector<long double> y_pred = predict_without_bias_append(X);

      long double rss = 0.0;
      for (int i = 0; i < n; i++) {
        long double error = y[i] - y_pred[i];
        rss += error * error;
      }

      long double w_norm_sq = 0.0;
      for (int i = 0; i + 1 < weights.size(); i++) {
        w_norm_sq += weights[i] * weights[i];
      }

      long double trace_Sigma = 0.0;
      for (int i = 0; i < Sigma.rows() - 1; i++) {
        trace_Sigma += Sigma.value[i][i];
      }

      long double gamma = (p - 1) - lambda * trace_Sigma;

      gamma = std::max((long double)0.0, gamma);
      gamma = std::min((long double)(p - 1), gamma);

      long double new_lambda = lambda;
      long double new_alpha = alpha;

      if (w_norm_sq > 1e-18) {
        new_lambda = gamma / w_norm_sq;
      }

      if (rss > 1e-18 && n > gamma) {
        new_alpha = (n - gamma) / rss;
      }

      if (new_lambda > 1e-18) {
        lambda = new_lambda;
      }

      if (new_alpha > 1e-18) {
        alpha = new_alpha;
      }

      long double diff = 0.0;
      for (int i = 0; i < p; i++) {
        diff += std::abs(weights[i] - old_weights[i]);
      }

      if (diff < tol) {
        break;
      }

      old_weights = weights;
    }
  }

  std::vector<long double> predict(std::vector<std::vector<long double>> X) override {
    for (std::vector<long double>& row : X) {
      row.push_back(1.0);
    }

    return predict_without_bias_append(X);
  }

  std::vector<long double> predict_without_bias_append(std::vector<std::vector<long double>> X) {
    Matlib::Matrix mX(X);
    Matlib::Matrix mW(weights);
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

  std::vector<std::vector<long double>> X_train = df_train.drop({"price"}).data;
  std::vector<std::vector<long double>> X_test = df_test.drop({"price"}).data;

  std::vector<long double> y_train = df_train.get_column_data("price");
  std::vector<long double> y_test = df_test.get_column_data("price");

  BayesianRidge model;
  model.fit(X_train, y_train);

  std::cerr << "Score: " << model.score(X_test, y_test) << std::endl;

  std::vector<long double> model_weights = model.get_model_weights();

  std::cerr << "Weights: [ ";
  for (size_t i = 0; i + 1 < model_weights.size(); i++) {
    std::cerr << model_weights[i] << " ";
  }

  std::cerr << "]; Bias = " << model_weights.back()
            << "; Alpha = " << model.alpha
            << "; Lambda = " << model.lambda
            << std::endl;
}