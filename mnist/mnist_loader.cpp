#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>

MNISTLoader::MNISTLoader(const std::string& dir) : data_dir(dir) {}

std::vector<float> MNISTLoader::read_images(const std::string& filename, int& num_images, int& rows, int& cols) {
    std::ifstream file(data_dir + "/" + filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    // 读取文件头
    int magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    
    if (magic_number != 2051) {
        throw std::runtime_error("无效的图像文件格式");
    }
    
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = __builtin_bswap32(num_images);
    
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = __builtin_bswap32(rows);
    
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cols = __builtin_bswap32(cols);
    
    // 读取图像数据
    int image_size = rows * cols;
    std::vector<float> images(num_images * image_size);
    
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i * image_size + j] = pixel / 255.0f; // 归一化到0-1
        }
    }
    
    return images;
}

std::vector<int> MNISTLoader::read_labels(const std::string& filename, int& num_labels) {
    std::ifstream file(data_dir + "/" + filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    // 读取文件头
    int magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    
    if (magic_number != 2049) {
        throw std::runtime_error("无效的标签文件格式");
    }
    
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels);
    
    // 读取标签数据
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }
    
    return labels;
}

void MNISTLoader::load_training_data(Tensor& images, Tensor& labels) {
    int num_images, rows, cols;
    std::vector<float> image_data = read_images("train-images-idx3-ubyte", num_images, rows, cols);
    
    int num_labels;
    std::vector<int> label_data = read_labels("train-labels-idx1-ubyte", num_labels);
    
    // 创建图像张量: [num_images, rows*cols]
    std::vector<size_t> image_shape = {static_cast<size_t>(num_images), static_cast<size_t>(rows * cols)};
    images = Tensor(ShapeTag{}, image_shape, DType::kFloat, DeviceType::kCPU);
    
    // 复制图像数据
    for (size_t i = 0; i < image_data.size(); ++i) {
        images.data<float>()[i] = image_data[i];
    }
    
    // 创建标签张量: [num_labels]
    std::vector<size_t> label_shape = {static_cast<size_t>(num_labels)};
    labels = Tensor(ShapeTag{}, label_shape, DType::kFloat, DeviceType::kCPU);
    
    // 复制标签数据
    for (size_t i = 0; i < label_data.size(); ++i) {
        labels.data<float>()[i] = static_cast<float>(label_data[i]);
    }
}

void MNISTLoader::load_test_data(Tensor& images, Tensor& labels) {
    int num_images, rows, cols;
    std::vector<float> image_data = read_images("t10k-images-idx3-ubyte", num_images, rows, cols);
    
    int num_labels;
    std::vector<int> label_data = read_labels("t10k-labels-idx1-ubyte", num_labels);
    
    // 创建图像张量: [num_images, rows*cols]
    std::vector<size_t> image_shape = {static_cast<size_t>(num_images), static_cast<size_t>(rows * cols)};
    images = Tensor(ShapeTag{}, image_shape, DType::kFloat, DeviceType::kCPU);
    
    // 复制图像数据
    for (size_t i = 0; i < image_data.size(); ++i) {
        images.data<float>()[i] = image_data[i];
    }
    
    // 创建标签张量: [num_labels]
    std::vector<size_t> label_shape = {static_cast<size_t>(num_labels)};
    labels = Tensor(ShapeTag{}, label_shape, DType::kFloat, DeviceType::kCPU);
    
    // 复制标签数据
    for (size_t i = 0; i < label_data.size(); ++i) {
        labels.data<float>()[i] = static_cast<float>(label_data[i]);
    }
}
