// Apache-2.0. Preserve the rank's OpenMP team across an operator-local override.
#pragma once
#include <cstdint>
extern "C" int64_t pegainfer_kutacc_set_num_threads(int64_t n);
class PegainferThreadScope {
    int64_t previous_ = pegainfer_kutacc_set_num_threads(0);
public:
    PegainferThreadScope() = default;
    PegainferThreadScope(const PegainferThreadScope &) = delete;
    PegainferThreadScope &operator=(const PegainferThreadScope &) = delete;
    ~PegainferThreadScope() { pegainfer_kutacc_set_num_threads(previous_); }
};
