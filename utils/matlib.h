#pragma once

#include <vector>
#include <stdexcept>

namespace Matlib {
  struct Matrix {
    std::vector<std::vector<long double>> value;

    Matrix() = default;

    Matrix(std::vector<long double> value) {
      this->value.assign(value.size(), std::vector<long double>(1));

      for (size_t i = 0; i < value.size(); i++) {
        this->value[i][0] = value[i];
      }
    }

    Matrix(std::vector<std::vector<long double>> value) {
      this->value = value;
    }
    
    Matrix(int h, int w, long double x = 0) {
      if(h < 1 || w < 1) {
        throw std::invalid_argument("Matrix construction(h, w, x = 0): h, w should be larger than 0");
      }

      value.assign(h, std::vector<long double>(w, x));
    }

    int rows() const {
      return static_cast<int>(value.size());
    }

    int cols() const {
      return value.empty() ? 0 : static_cast<int>(value[0].size());
    }
  };

  Matrix add(Matrix A, const Matrix B);
  Matrix subtract(const Matrix A, const Matrix B);
  Matrix transpose(const Matrix A);
  Matrix scalar_multiply(Matrix A, const long double x);
  Matrix matrix_multiply(const Matrix A, const Matrix B);
  Matrix inverse(Matrix A);
}