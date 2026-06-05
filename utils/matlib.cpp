#include "matlib.h"
#include <vector>
#include <stdexcept>

namespace Matlib {
  Matrix add(Matrix a, const Matrix b) {
    if(a.rows() != b.rows() || a.cols() != b.cols()) {
      throw std::invalid_argument("Matrix add: two matrix should have the same dimension");
    }

    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        a.value[i][j] += b.value[i][j];
      }
    }

    return a;
  }
  Matrix subtract(Matrix a, const Matrix b) {
    if(a.rows() != b.rows() || a.cols() != b.cols()) {
      throw std::invalid_argument("Matrix add: two matrix should have the same dimension");
    }

    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        a.value[i][j] -= b.value[i][j];
      }
    }

    return a;
  }
  Matrix transpose(const Matrix a) {
    Matrix b(a.cols(), a.rows());

    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        b.value[j][i] = a.value[i][j];
      }
    }

    return b;
  }
  Matrix scalar_multiply(Matrix a, const long double x) {
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        a.value[i][j] *= x;
      }
    }
    return a;
  }
  Matrix matrix_multiply(const Matrix a, const Matrix b) {
    if(a.cols() != b.rows()) {
      throw std::invalid_argument("Matrix multiplication: Matrix dimensions do not match for multiplication");
    }

    Matrix c(a.rows(), b.cols(), 0);

    for (int i = 0; i < a.rows(); i++) {
      for (int k = 0; k < a.cols(); k++) {
        for (int j = 0; j < b.cols(); j++) {
          c.value[i][j] += a.value[i][k] * b.value[k][j];
        }
      }
    }

    return c;
  }
}