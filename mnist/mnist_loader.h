#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "../include/Tensor.h"
#include <filesystem>
#include <string>
#include <vector>

class MNISTLoader {
private:
    std::filesystem::path data_dir;
    DeviceType device;

    // 将 filename 解析为 data_dir 下的安全绝对路径，阻止路径穿越
    std::filesystem::path safe_path(const std::string& filename) const;

    // 读取MNIST图像文件
    std::vector<float> read_images(const std::string& filename, int& num_images, int& rows, int& cols);

    // 读取MNIST标签文件
    std::vector<int> read_labels(const std::string& filename, int& num_labels);
    
public:
    MNISTLoader(const std::string& dir, DeviceType dev = DeviceType::kMPS);
    
    // 加载训练数据
    void load_training_data(Tensor& images, Tensor& labels);
    
    // 加载测试数据
    void load_test_data(Tensor& images, Tensor& labels);
};

#endif // MNIST_LOADER_H
