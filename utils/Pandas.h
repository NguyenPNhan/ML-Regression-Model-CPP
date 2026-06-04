#include <vector>
#include <string>


namespace Pandas {
  struct DataFrame;

  DataFrame read_csv(const std::string path);
}