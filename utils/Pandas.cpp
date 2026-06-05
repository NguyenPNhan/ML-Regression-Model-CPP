#include "pandas.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

namespace Pandas {
  DataFrame read_csv(const std::string path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
      throw std::runtime_error("read_csv: File does not exist or cannot be open: " + path);
    }
    
    std::string str, token;
    std::getline(input_file, str);

    std::stringstream string_stream(str);
    std::vector<std::string>columns;
    
    DataFrame df;

    while (std::getline(string_stream, token, ',')) {
      df.columns.emplace_back(token);
    }

    while (std::getline(input_file, str)) {
      string_stream.clear();
      string_stream.str(str);

      df.data.emplace_back(std::vector<long double>(df.columns.size()));

      for (int i = 0; std::getline(string_stream, token, ','); i++) {
        df.data.back()[i] = std::stold(token);
      }
    }

    return df;
  }
}