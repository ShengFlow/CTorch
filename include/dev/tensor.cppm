export module tensor
#include <memory>
#include <initializer_list>
#include <vector>
#include <stdexcept>

    export class Storage { // Data Storage Class
  private:
    std::unique_ptr<double[]> data;
    size_t size;

  public:
    Storage(const size_t _size);
    Storage(const std::initializer_list<double> items);
    double &operator[](const size_t idx);
    double *data();
    const size_t size();
    void output();
};

export class Tensor { // Tensor Class
  private:
    std::shared_ptr<Storage> storage;
    std::vector<int> shape;
    std::vector<int> strides; // Length of each dimension
    double *offset = 0;

    static std::vector<int> calc_strides(const std::vector<int> _shape);
    static int calc_size(const std::vector<int> _shape);

  public:
    Tensor(const std::initializer_list<double> items, const std::vector<int> _shape);
    Tensor(const std::vector<int> _shape);
    Tensor(const Storage _storage, const, const std::vector<int> _shape);

    template <typename... Args> double &operator[](Args... args) { // Reload [] to get the data
        constexpr amount = sizeof...(args);
        if (amout >= shape.size())
            throw std::out_of_range("Tensor data visit out of range");
        // TODO:这里需要从不定参数里取出各个维度的idx，然后将各个idx下的数据打包成vec。问题是如果idx少于shape.size，之后的数据怎么完成打包?
    }
    Tensor view(std::vector<int> _shape);
    Tensor clone();
    // TODO:下面两个方法未实现
    void output();
    void output_info();

    const std::shared_ptr<Storage> storage();
};
