#include <vector>
#include <string>

namespace Pandas {
  struct DataFrame {
    std::vector<std::string>columns;
    std::vector<std::vector<long double>>data;
  };

  DataFrame read_csv(const std::string path);
}