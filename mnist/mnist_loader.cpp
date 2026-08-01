#include "mnist_loader.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace {

constexpr int kMaxMNISTDim = 10000;                 // 单张图像最大边长
constexpr long long kMaxTotalElements = 1LL << 30;  // 最大总元素数（约 4 GB float）
constexpr long long kMaxTotalLabels = 1LL << 30;    // 最大标签数

// 安全乘法：result = a * b，溢出或超过 limit 时返回 false
bool safe_mul(long long a, long long b, long long limit, long long& result) {
    if (a <= 0 || b <= 0) return false;
    if (a > limit / b) return false;
    result = a * b;
    return result <= limit;
}

// 校验最终解析路径是否严格位于 base 目录下
bool is_under_base(const std::filesystem::path& base, const std::filesystem::path& target) {
    auto base_abs = base.lexically_normal();
    auto target_abs = target.lexically_normal();

    auto base_str = base_abs.string();
    auto target_str = target_abs.string();

    // 统一追加分隔符后再比较，防止 /data/foo 匹配 /data/foobar
    if (!base_str.empty() && base_str.back() != std::filesystem::path::preferred_separator) {
        base_str.push_back(std::filesystem::path::preferred_separator);
    }
    return target_str.size() > base_str.size() && target_str.compare(0, base_str.size(), base_str) == 0;
}

} // namespace

MNISTLoader::MNISTLoader(const std::string& dir, DeviceType dev) : device(dev) {
    std::error_code ec;
    auto canonical = std::filesystem::canonical(dir, ec);
    if (ec) {
        throw std::runtime_error("无法解析 MNIST 数据目录: " + dir);
    }
    data_dir = canonical;
}

std::filesystem::path MNISTLoader::safe_path(const std::string& filename) const {
    if (filename.empty() || filename.find('\0') != std::string::npos) {
        throw std::runtime_error("非法文件名");
    }

    std::error_code ec;
    std::filesystem::path target = data_dir / filename;
    std::filesystem::path resolved;

    if (std::filesystem::exists(target, ec)) {
        resolved = std::filesystem::canonical(target, ec);
        if (ec) {
            throw std::runtime_error("无法解析目标文件: " + target.string());
        }
    } else {
        resolved = target;
    }

    if (!is_under_base(data_dir, resolved)) {
        throw std::runtime_error("路径穿越被阻止: " + filename);
    }

    return resolved;
}

std::vector<float> MNISTLoader::read_images(const std::string& filename, int& num_images, int& rows, int& cols) {
    auto path = safe_path(filename);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + path.string());
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

    // 安全校验
    if (num_images <= 0 || rows <= 0 || cols <= 0) {
        throw std::runtime_error("MNIST 文件头包含非正维度");
    }
    if (rows > kMaxMNISTDim || cols > kMaxMNISTDim) {
        throw std::runtime_error("MNIST 图像尺寸超过允许上限");
    }

    long long image_size_ll = 0;
    if (!safe_mul(rows, cols, kMaxTotalElements, image_size_ll)) {
        throw std::runtime_error("MNIST rows*cols 乘法溢出或超过上限");
    }

    long long total_ll = 0;
    if (!safe_mul(num_images, image_size_ll, kMaxTotalElements, total_ll)) {
        throw std::runtime_error("MNIST num_images*image_size 乘法溢出或超过上限");
    }

    // 校验文件实际大小是否足以容纳声明的数据
    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if (!ec) {
        constexpr long long header_size = 16; // 4 个 int32
        long long expected_payload = total_ll;            // 每个像素 1 字节
        long long expected_min_size = header_size + expected_payload;
        if (file_size < static_cast<uintmax_t>(expected_min_size)) {
            throw std::runtime_error("MNIST 文件大小与头声明不一致（可能数据不足）");
        }
    }

    // 读取图像数据
    std::vector<float> images(static_cast<size_t>(total_ll));
    const auto image_size = static_cast<size_t>(image_size_ll);

    for (int i = 0; i < num_images; ++i) {
        for (size_t j = 0; j < image_size; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[static_cast<size_t>(i) * image_size + j] = static_cast<float>(pixel) / 255.0f;
        }
    }

    return images;
}

std::vector<int> MNISTLoader::read_labels(const std::string& filename, int& num_labels) {
    auto path = safe_path(filename);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + path.string());
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

    if (num_labels <= 0) {
        throw std::runtime_error("MNIST 标签数非正");
    }
    if (num_labels > kMaxTotalLabels) {
        throw std::runtime_error("MNIST 标签数超过允许上限");
    }

    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if (!ec) {
        constexpr long long header_size = 8; // 2 个 int32
        long long expected_min_size = header_size + static_cast<long long>(num_labels);
        if (file_size < static_cast<uintmax_t>(expected_min_size)) {
            throw std::runtime_error("MNIST 标签文件大小与头声明不一致");
        }
    }

    std::vector<int> labels(static_cast<size_t>(num_labels));
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

void MNISTLoader::load_training_data(Tensor& images, Tensor& labels) {
    int num_images, rows, cols;
    std::vector<float> image_data = read_images("train-images-idx3-ubyte", num_images, rows, cols);

    int num_labels;
    std::vector<int> label_data = read_labels("train-labels-idx1-ubyte", num_labels);

    if (num_images != num_labels) {
        throw std::runtime_error("MNIST 训练集图像数与标签数不一致");
    }

    std::vector<size_t> image_shape = {static_cast<size_t>(num_images), static_cast<size_t>(rows) * static_cast<size_t>(cols)};
    images = Tensor(ShapeTag{}, image_shape, DType::kFloat, device);

    float* img_ptr = images.data_write<float>();
    for (size_t i = 0; i < image_data.size(); ++i) {
        img_ptr[i] = image_data[i];
    }

    std::vector<size_t> label_shape = {static_cast<size_t>(num_labels)};
    labels = Tensor(ShapeTag{}, label_shape, DType::kFloat, device);

    for (size_t i = 0; i < label_data.size(); ++i) {
        labels.data_write<float>()[i] = static_cast<float>(label_data[i]);
    }
}

void MNISTLoader::load_test_data(Tensor& images, Tensor& labels) {
    int num_images, rows, cols;
    std::vector<float> image_data = read_images("t10k-images-idx3-ubyte", num_images, rows, cols);

    int num_labels;
    std::vector<int> label_data = read_labels("t10k-labels-idx1-ubyte", num_labels);

    if (num_images != num_labels) {
        throw std::runtime_error("MNIST 测试集图像数与标签数不一致");
    }

    std::vector<size_t> image_shape = {static_cast<size_t>(num_images), static_cast<size_t>(rows) * static_cast<size_t>(cols)};
    images = Tensor(ShapeTag{}, image_shape, DType::kFloat, device);

    for (size_t i = 0; i < image_data.size(); ++i) {
        images.data_write<float>()[i] = image_data[i];
    }

    std::vector<size_t> label_shape = {static_cast<size_t>(num_labels)};
    labels = Tensor(ShapeTag{}, label_shape, DType::kFloat, device);

    for (size_t i = 0; i < label_data.size(); ++i) {
        labels.data_write<float>()[i] = static_cast<float>(label_data[i]);
    }
}
