#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace Pandas {
  struct DataFrame {
    std::vector<std::string> columns;
    std::vector<std::vector<long double>> data;

    std::vector<long double> get_column_data(std::string column) {
      std::vector<std::string>::iterator column_iterator = std::find(columns.begin(), columns.end(), column);
      if (column_iterator != columns.end()) {
        throw std::invalid_argument("DataFrame::get_column_data(): column '" + column + "' not found");
      }

      int column_index = column_iterator - columns.begin();
      return data[column_index];
    }
    
    DataFrame drop(std::vector<std::string>drop_columns) {
      std::unordered_set<std::string> drop_set(drop_columns.begin(), drop_columns.end());

      DataFrame df_result;
      std::vector<size_t> keep_indices;

      for (size_t i = 0; i < columns.size(); i++) {
        if (!drop_set.count(columns[i])) {
          keep_indices.emplace_back(i);
          df_result.columns.emplace_back(columns[i]);
        }
      }

      df_result.data.reserve(data.size());

      for (const std::vector<long double>& row : data) {
        std::vector<long double> new_row;
        new_row.reserve(keep_indices.size());
        
        for (size_t& index : keep_indices) {
          new_row.emplace_back(data[index]);
        }

        df_result.data.emplace_back(new_row);
      }

      return df_result;
    }
  };

  DataFrame read_csv(const std::string path);
}