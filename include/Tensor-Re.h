//
// Created by 于梁 on 2026/4/5.
//

#ifndef CTORCH_TENSOR_RE_H
#define CTORCH_TENSOR_RE_H

#include "Ctools.h"
#include <vector>

class Tensor {
    std::vector<size_t> computeStrides(const std::vector<size_t> &shape);
};

#endif // CTORCH_TENSOR_RE_H