#include <vector>
#include <stdexcept>

namespace Matlib {
  struct Matrix {
    std::vector<std::vector<long double>>value;

    Matrix() = default;
    
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

  Matrix add(Matrix a, const Matrix b);
  Matrix subtract(const Matrix a, const Matrix b);
  Matrix transpose(const Matrix a);
  Matrix scalar_multiply(Matrix a, const long double x);
  Matrix matrix_multiply(const Matrix a, const Matrix b);
}