//
// Created by Beapoe on 2025/8/30.
//
import Tensor_dev;
#include <cassert>
#include <vector>

int main() {
    Tensor a(0.0f);
    assert(a.dtype() == DType::kFloat);

    Tensor b({1.0f,2.0f});
    assert(b.numel() == 2);

    Tensor c({true,false,true});
    assert(c.dtype() == DType::kBool);

    auto tag = ShapeTag();
    Tensor d(tag,{2,2});
    assert(d.shape() == std::vector<size_t>({2,2}) && d.data() == 0);

    Tensor e({2.0f,3.0f},{2,1});
    assert(e.numel() == 2 && e.dtype() == DType::kFloat);

    Tensor f(e);
    assert(f.numel() == 2 && e.dtype() == DType::kFloat);

    Tensor g(std::move(f));
    assert(g.numel() == 2 && g.dtype() == DType::kFloat);

    return 0;
}