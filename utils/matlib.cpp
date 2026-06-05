#include "matlib.h"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace Matlib {
  Matrix add(Matrix A, const Matrix B) {
    if(A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrix add: two matrix should have the same dimension");
    }

    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < A.cols(); j++) {
        A.value[i][j] += B.value[i][j];
      }
    }

    return A;
  }
  
  Matrix subtract(Matrix A, const Matrix B) {
    if(A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrix add: two matrix should have the same dimension");
    }

    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < A.cols(); j++) {
        A.value[i][j] -= B.value[i][j];
      }
    }

    return A;
  }

  Matrix transpose(const Matrix A) {
    Matrix B(A.cols(), A.rows());

    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < A.cols(); j++) {
        B.value[j][i] = A.value[i][j];
      }
    }

    return B;
  }

  Matrix scalar_multiply(Matrix A, const long double x) {
    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < A.cols(); j++) {
        A.value[i][j] *= x;
      }
    }
    return A;
  }

  Matrix matrix_multiply(const Matrix A, const Matrix B) {
    if(A.cols() != B.rows()) {
      throw std::invalid_argument("Matrix multiplication: Matrix dimensions do not match for multiplication");
    }

    Matrix C(A.rows(), B.cols(), 0);

    for (int i = 0; i < A.rows(); i++) {
      for (int k = 0; k < A.cols(); k++) {
        for (int j = 0; j < B.cols(); j++) {
          C.value[i][j] += A.value[i][k] * B.value[k][j];
        }
      }
    }

    return C;
  }

  Matrix inverse(Matrix A) {
    if (A.rows() != A.cols()) {
      throw std::invalid_argument("Matrix inverse: Matrix should be square matrix");
    }

    int n = A.rows();

    Matrix I(n, n);
    for (int i = 0; i < n; i++) {
      I.value[i][i] = 1.0;
    }

    for (int i = 0; i < n; i++) {
      long double pivot = A.value[i][i];

      if (std::fabs(pivot) < 1e-12) {
        throw std::runtime_error("Matrix inverse: Matrix is singular, cannot inverse");
      }

      for (int j = 0; j < n; j++) {
        A.value[i][j] /= pivot;
        I.value[i][j] /= pivot;
      }

      for (int r = 0; r < n; r++) {
        if (r == i) {
          continue;
        }

        long double factor = A.value[r][i];

        for (int c = 0; c < n; c++) {
          A.value[r][c] -= factor * A.value[i][c];
          I.value[r][c] -= factor * I.value[i][c];
        }
      }
    }

    return I;
  }
}