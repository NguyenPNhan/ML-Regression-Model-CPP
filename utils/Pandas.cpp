#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

namespace Pandas {
  struct DataFrame {
    std::vector<std::string>columns;
    std::vector<std::vector<long double>>data;
  };

  DataFrame read_csv(const std::string path) {
    std::ifstream inputFile(path);
    
    std::string str, token;
    std::getline(inputFile, str);

    std::stringstream strStream(str);
    std::vector<std::string>columns;
    
    DataFrame df;

    while (std::getline(strStream, token, ',')) {
      df.columns.emplace_back(token);
    }

    while (!inputFile.eof()) {
      std::getline(inputFile, str);

      strStream.clear();
      strStream.str(str);

      df.data.emplace_back(std::vector<long double>(df.columns.size()));
      for (int i = 0; std::getline(strStream, token, ','); i++) {
        df.data.back()[i] = std::stold(token);
      }
    }
  }
}