#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "Tensor.h"
#include <string>
#include <vector>

class MNISTLoader {
private:
    std::string data_dir;
    
    // 读取MNIST图像文件
    std::vector<float> read_images(const std::string& filename, int& num_images, int& rows, int& cols);
    
    // 读取MNIST标签文件
    std::vector<int> read_labels(const std::string& filename, int& num_labels);
    
public:
    MNISTLoader(const std::string& dir);
    
    // 加载训练数据
    void load_training_data(Tensor& images, Tensor& labels);
    
    // 加载测试数据
    void load_test_data(Tensor& images, Tensor& labels);
};

#endif // MNIST_LOADER_H
