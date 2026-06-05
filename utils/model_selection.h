#include "pandas.h"
#include <utility>

std::pair<Pandas::DataFrame, Pandas::DataFrame>train_test_split(const Pandas::DataFrame df, double test_size, int random_state);