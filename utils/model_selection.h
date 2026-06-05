#pragma once

#include "pandas.h"
#include <utility>

std::pair<Pandas::DataFrame, Pandas::DataFrame> train_test_split(const Pandas::DataFrame& df, double test_size = 0.2, int random_state = 42);