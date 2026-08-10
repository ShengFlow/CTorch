// (SECURITY_SENSITIVE) MNIST Loader 安全回归测试
// 验证路径穿越、整数溢出、文件头与大小不一致等恶意输入被正确拦截。

#include "mnist/mnist_loader.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static void write_int32_be(std::ofstream& file, int32_t value) {
    uint32_t be = static_cast<uint32_t>(__builtin_bswap32(value));
    file.write(reinterpret_cast<const char*>(&be), sizeof(be));
}

static void write_valid_labels(const fs::path& base) {
    fs::path lbl_path = base / "train-labels-idx1-ubyte";
    std::ofstream lbl(lbl_path, std::ios::binary);
    write_int32_be(lbl, 2049); // magic
    write_int32_be(lbl, 2);    // num_labels
    for (int i = 0; i < 2; ++i) {
        unsigned char label = static_cast<unsigned char>(i);
        lbl.write(reinterpret_cast<const char*>(&label), sizeof(label));
    }
}

static bool expect_runtime_error(const std::string& test_name,
                                 const std::function<void()>& fn,
                                 const std::string& expected_substring) {
    try {
        fn();
        std::cerr << "[-] " << test_name << "：未抛出异常\n";
        return false;
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find(expected_substring) != std::string::npos) {
            std::cout << "[+] " << test_name << " 通过\n";
            return true;
        }
        std::cerr << "[-] " << test_name << "：异常不匹配: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[-] " << test_name << "：异常类型不匹配: " << e.what() << "\n";
        return false;
    }
}

static bool test_path_traversal_blocked() {
    fs::path base = fs::temp_directory_path() / "ctorch_mnist_sec_base";
    fs::path outside = fs::temp_directory_path() / "ctorch_mnist_secret.txt";
    fs::remove_all(base);
    fs::create_directories(base);
    {
        std::ofstream secret(outside);
        secret << "SECRET";
    }

    // 在 base 内创建指向外部文件的符号链接
    fs::path link_target = base / "train-images-idx3-ubyte";
    fs::create_symlink(outside, link_target);
    write_valid_labels(base);

    MNISTLoader loader(base.string(), DeviceType::kCPU);
    Tensor images, labels;
    return expect_runtime_error("路径穿越",
                                [&]() { loader.load_training_data(images, labels); },
                                "路径穿越被阻止");
}

static bool test_integer_overflow_blocked() {
    fs::path base = fs::temp_directory_path() / "ctorch_mnist_sec_base";
    fs::remove_all(base);
    fs::create_directories(base);

    fs::path img_path = base / "train-images-idx3-ubyte";
    std::ofstream file(img_path, std::ios::binary);
    write_int32_be(file, 2051);     // magic
    write_int32_be(file, 1);        // num_images
    write_int32_be(file, 100000);   // rows
    write_int32_be(file, 100000);   // cols
    // 不写入像素数据
    file.close();
    write_valid_labels(base);

    MNISTLoader loader(base.string(), DeviceType::kCPU);
    Tensor images, labels;
    return expect_runtime_error("整数溢出/超大尺寸",
                                [&]() { loader.load_training_data(images, labels); },
                                "超过允许上限");
}

static bool test_file_size_mismatch_blocked() {
    fs::path base = fs::temp_directory_path() / "ctorch_mnist_sec_base";
    fs::remove_all(base);
    fs::create_directories(base);

    fs::path img_path = base / "train-images-idx3-ubyte";
    std::ofstream file(img_path, std::ios::binary);
    write_int32_be(file, 2051); // magic
    write_int32_be(file, 10);   // num_images
    write_int32_be(file, 28);   // rows
    write_int32_be(file, 28);   // cols
    // 不写入像素数据
    file.close();
    write_valid_labels(base);

    MNISTLoader loader(base.string(), DeviceType::kCPU);
    Tensor images, labels;
    return expect_runtime_error("文件大小不一致",
                                [&]() { loader.load_training_data(images, labels); },
                                "文件大小与头声明不一致");
}

static bool test_valid_file_loads() {
    try {
        fs::path base = fs::temp_directory_path() / "ctorch_mnist_sec_base";
        fs::remove_all(base);
        fs::create_directories(base);

        fs::path img_path = base / "train-images-idx3-ubyte";
        std::ofstream img(img_path, std::ios::binary);
        write_int32_be(img, 2051); // magic
        write_int32_be(img, 2);    // num_images
        write_int32_be(img, 2);    // rows
        write_int32_be(img, 2);    // cols
        for (int i = 0; i < 2 * 2 * 2; ++i) {
            unsigned char pixel = static_cast<unsigned char>(i * 10);
            img.write(reinterpret_cast<const char*>(&pixel), sizeof(pixel));
        }
        img.close();
        write_valid_labels(base);

        MNISTLoader loader(base.string(), DeviceType::kCPU);
        Tensor images, labels;
        loader.load_training_data(images, labels);

        if (images.numel() != 8 || labels.numel() != 2) {
            std::cerr << "[-] 正常加载测试：维度或数量不匹配\n";
            return false;
        }

        std::cout << "[+] 正常 MNIST 文件加载成功\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[-] 正常加载测试异常: " << e.what() << "\n";
        return false;
    }
}

int main() {
    bool ok = true;
    ok &= test_path_traversal_blocked();
    ok &= test_integer_overflow_blocked();
    ok &= test_file_size_mismatch_blocked();
    ok &= test_valid_file_loads();

    if (ok) {
        std::cout << "\n[OK] MNIST Loader 安全回归测试全部通过\n";
        return 0;
    }
    std::cerr << "\n[FAIL] MNIST Loader 安全回归测试存在失败项\n";
    return 1;
}
