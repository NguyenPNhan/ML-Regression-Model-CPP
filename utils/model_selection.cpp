#include "pandas.h"
#include <utility>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>

std::pair<Pandas::DataFrame, Pandas::DataFrame>train_test_split(const Pandas::DataFrame& df, double test_size = 0.2, int random_state = 42) {
  if(test_size <= 0.0 || test_size >= 1.0) {
    throw std::invalid_argument("train_test_split: test_size must be between 0 and 1");
  }

  std::mt19937 rng(random_state);

  size_t n = df.data.size();
  size_t n_test = static_cast<int>(std::floor(n * test_size));
  size_t n_train = n - n_test;

  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);

  Pandas::DataFrame df_train, df_test;
  df_train.columns = df_test.columns = df.columns;

  df_train.data.reserve(n_train);
  df_test.data.reserve(n_test);

  for (size_t i = 0; i < n_train; i++) {
    df_train.data.emplace_back(df.data[perm[i]]);
  }
  for (size_t i = n_train; i < n; i++) {
    df_test.data.emplace_back(df.data[perm[i]]);
  }

  return std::make_pair(df_train, df_test);
}