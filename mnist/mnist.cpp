#include "mnist_loader.h"
#include "AutoDiff.h"
#include "Ctorch_Error.h"
#include <iostream>
#include <chrono>
#include <iomanip>

// 简单的全连接神经网络类
class NeuralNetwork {
private:
    Tensor W1, b1, W2, b2;
    float learning_rate;
    
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, float lr) : 
        learning_rate(lr) {
        
        // 初始化权重和偏置 - 对于线性模型，我们只需要W2和b2
        // 注意：hidden_size参数在这里被忽略，因为我们使用的是线性模型
        std::vector<size_t> w2_shape = {static_cast<size_t>(input_size), static_cast<size_t>(output_size)};
        std::vector<size_t> b2_shape = {static_cast<size_t>(output_size)};
        
        // 使用随机初始化
        // 对于线性模型，我们不需要W1和b1
        W2 = Tensor(ShapeTag{}, w2_shape, DType::kFloat, DeviceType::kCPU);
        b2 = Tensor(ShapeTag{}, b2_shape, DType::kFloat, DeviceType::kCPU);
        
        // 使用Xavier初始化，适合线性模型
        float std = sqrt(1.0f / input_size);
        for (size_t i = 0; i < W2.numel(); ++i) {
            // 生成[-1, 1]的随机数，然后乘以标准差
            float rand_val = (2.0f * rand() / static_cast<float>(RAND_MAX)) - 1.0f;
            W2.data<float>()[i] = rand_val * std;
        }
        
        // 偏置初始化为0
        for (size_t i = 0; i < b2.numel(); ++i) {
            b2.data<float>()[i] = 0.0f;
        }
        
        // 注意：requires_grad将在train_step中设置，确保在AutoDiff上下文内部
    }
    
    // 前向传播
    Tensor forward(const Tensor& x) {
        // 简单的线性模型：x * W2 + b2
        Tensor z2 = x.matmul(W2);
        Tensor z3 = z2 + b2;
        
        // 直接返回 logits
        return z3;
    }
    
    // 训练一步
    float train_step(const Tensor& x, const Tensor& y) {
        // 打印当前函数名，方便调试
        std::cout << "=== Entering train_step ===" << std::endl;
        
        // 打印W2和b2的ID，确保它们在每次迭代中保持一致
        std::cout << "W2 ID: " << W2.id() << ", b2 ID: " << b2.id() << std::endl;
        
        // 检查AutoDiff上下文是否存在
        std::cout << "Checking AutoDiff context..." << std::endl;
        if (AutoDiffContext::current()) {
            std::cout << "AutoDiff context exists." << std::endl;
        } else {
            std::cout << "ERROR: AutoDiff context does not exist!" << std::endl;
        }
        
        // 重新设置requires_grad，确保在新的上下文中医正确跟踪
        std::cout << "Setting requires_grad(true) for W2 and b2..." << std::endl;
        W2.requires_grad(true);
        b2.requires_grad(true);
        
        // 再次检查AutoDiff上下文
        if (AutoDiffContext::current()) {
            std::cout << "AutoDiff context still exists after requires_grad." << std::endl;
        } else {
            std::cout << "ERROR: AutoDiff context lost after requires_grad!" << std::endl;
        }
        
        // 打印W2和b2的requires_grad状态
        std::cout << "W2 requires_grad: " << W2.requires_grad() << ", b2 requires_grad: " << b2.requires_grad() << std::endl;
        
        // 计算MSE损失 - 使用张量操作
        std::cout << "Computing forward pass..." << std::endl;
        Tensor y_pred = forward(x);
        std::cout << "Forward pass completed. y_pred ID: " << y_pred.id() << std::endl;
        
        // 首先将标签转换为one-hot编码
        Tensor y_one_hot = Tensor(ShapeTag{}, {static_cast<size_t>(y.numel()), 10}, DType::kFloat, DeviceType::kCPU);
        for (size_t i = 0; i < y.numel(); ++i) {
            int label = static_cast<int>(y.data<float>()[i]);
            for (size_t j = 0; j < 10; ++j) {
                y_one_hot.data<float>()[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
            }
        }
        
        // 计算MSE损失
        std::cout << "Computing loss..." << std::endl;
        Tensor diff = y_pred - y_one_hot;
        std::cout << "diff ID: " << diff.id() << std::endl;
        Tensor diff_squared = diff * diff;
        std::cout << "diff_squared ID: " << diff_squared.id() << std::endl;
        Tensor sum_diff = diff_squared.sum();
        std::cout << "sum_diff ID: " << sum_diff.id() << std::endl;
        float numel = static_cast<float>(diff_squared.numel());
        Tensor loss = sum_diff / numel;
        std::cout << "loss ID: " << loss.id() << std::endl;
        float loss_value = loss.item<float>();
        
        // 打印损失值
        std::cout << "Loss: " << loss_value << std::endl;
        
        // 反向传播
        std::cout << "Starting backward propagation..." << std::endl;
        backward(loss);
        std::cout << "Backward propagation completed." << std::endl;
        
        // 打印梯度信息
        Tensor W2_grad = grad(W2);
        std::cout << "W2 grad shape: ";
        for (size_t s : W2_grad.sizes()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        std::cout << "W2 grad[0][0]: " << W2_grad.data<float>()[0] << std::endl;
        
        // 打印W2和b2的requires_grad状态
        std::cout << "W2 requires_grad: " << W2.requires_grad() << ", b2 requires_grad: " << b2.requires_grad() << std::endl;
        
        // 更新参数
        update_parameters();
        
        std::cout << "=== Exiting train_step ===" << std::endl;
        return loss_value;
    }
    
    // 更新参数
    void update_parameters() {
        // 使用真实的梯度信息进行参数更新
        float lr = learning_rate;
        
        // 获取梯度并更新参数
        if (W2.requires_grad()) {
            Tensor W2_grad = grad(W2);
            std::cout << "W2 Grad Shape: ";
            for (size_t s : W2_grad.sizes()) {
                std::cout << s << " ";
            }
            std::cout << std::endl;
            
            // 获取W2的数据指针
            float* W2_data = W2.data<float>();
            const float* W2_grad_data = W2_grad.data<float>();
            size_t numel = W2.numel();
            
            // 打印W2的第一个元素和对应的梯度
            std::cout << "W2[0] before: " << W2_data[0] << ", grad: " << W2_grad_data[0] << std::endl;
            
            // 更新权重
            for (size_t i = 0; i < numel; ++i) {
                W2_data[i] -= W2_grad_data[i] * lr;
            }
            
            // 打印W2的第一个元素更新后的值
            std::cout << "W2[0] after: " << W2_data[0] << std::endl;
        }
        
        if (b2.requires_grad()) {
            Tensor b2_grad = grad(b2);
            float* b2_data = b2.data<float>();
            const float* b2_grad_data = b2_grad.data<float>();
            size_t numel = b2.numel();
            for (size_t i = 0; i < numel; ++i) {
                b2_data[i] -= b2_grad_data[i] * lr;
            }
        }
    }
    
    // 预测
    Tensor predict(const Tensor& x) {
        Tensor logits = forward(x);
        // 对 logits 应用 softmax 得到预测概率
        Tensor y_pred = logits.softmax(1);
        return y_pred;
    }
};

// 计算准确率
float calculate_accuracy(const Tensor& y_pred, const Tensor& y_true) {
    int correct = 0;
    int total = y_true.shape()[0];
    
    for (int i = 0; i < total; ++i) {
        // 找到预测的最大值索引
        int pred_label = 0;
        float max_prob = -1.0f;
        for (int j = 0; j < 10; ++j) {
            float prob = y_pred.data<float>()[i * 10 + j];
            if (prob > max_prob) {
                max_prob = prob;
                pred_label = j;
            }
        }
        
        // 比较预测标签和真实标签
        int true_label = static_cast<int>(y_true.data<float>()[i]);
        if (pred_label == true_label) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / total;
}

// 批次处理
void get_batch(const Tensor& images, const Tensor& labels, int batch_size, int batch_idx, Tensor& batch_images, Tensor& batch_labels) {
    int start = batch_idx * batch_size;
    int end = std::min(start + batch_size, static_cast<int>(images.shape()[0]));
    int actual_batch_size = end - start;
    
    // 创建批次图像张量
    std::vector<size_t> image_shape = {static_cast<size_t>(actual_batch_size), images.shape()[1]};
    batch_images = Tensor(ShapeTag{}, image_shape, DType::kFloat, DeviceType::kCPU);
    
    // 创建批次标签张量
    std::vector<size_t> label_shape = {static_cast<size_t>(actual_batch_size)};
    batch_labels = Tensor(ShapeTag{}, label_shape, DType::kFloat, DeviceType::kCPU);
    
    // 复制数据
    for (int i = 0; i < actual_batch_size; ++i) {
        for (size_t j = 0; j < images.shape()[1]; ++j) {
            float value = images.data<float>()[(start + i) * images.shape()[1] + j];
            batch_images.data<float>()[i * images.shape()[1] + j] = value;
        }
        
        float label = labels.data<float>()[start + i];
        batch_labels.data<float>()[i] = label;
    }
}

// 进度条显示函数
void show_progress(int current, int total, const std::string& status, float loss = 0.0f) {
    const int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    
    std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100 << "% "
              << status;
    
    if (loss > 0) {
        std::cout << " Loss: " << std::fixed << std::setprecision(4) << loss;
    }
    
    std::cout << std::flush;
}

int main() {
    try {
        // 设置输出级别为MINIUM，以取消TRACE和DEBUG级别的输出
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
        
        // 设置随机种子
        srand(42);
        
        // 加载MNIST数据
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "加载MNIST数据...");
        MNISTLoader loader(".");
        
        Tensor train_images, train_labels;
        Tensor test_images, test_labels;
        
        loader.load_training_data(train_images, train_labels);
        loader.load_test_data(test_images, test_labels);
        
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "数据加载完成");
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "训练集大小: " + std::to_string(train_images.shape()[0]));
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "测试集大小: " + std::to_string(test_images.shape()[0]));
        
        // 初始化神经网络
        int input_size = 784;  // 28x28
        int hidden_size = 128;
        int output_size = 10;   // 10个类别
        float learning_rate = 0.001;
        
        NeuralNetwork model(input_size, hidden_size, output_size, learning_rate);
        
        // 训练参数
        int epochs = 10;
        int batch_size = 64;
        int num_batches = train_images.shape()[0] / batch_size;
        
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "开始训练...");
        
        // 训练循环
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (int batch = 0; batch < num_batches; ++batch) {
                // 为每个批次创建新的AutoDiff上下文，与main.cpp中的方式一致
                AutoDiff autodiff;
                AutoDiffContext::Guard guard(&autodiff);
                
                // 获取批次数据
                Tensor batch_images, batch_labels;
                get_batch(train_images, train_labels, batch_size, batch, batch_images, batch_labels);
                
                // 设置requires_grad
                batch_images.requires_grad(false);
                batch_labels.requires_grad(false);
                
                // 训练一步
                float batch_loss = model.train_step(batch_images, batch_labels);
                epoch_loss += batch_loss;
                
                // 显示进度条
                std::string status = "Epoch " + std::to_string(epoch+1) + "/" + std::to_string(epochs);
                show_progress(batch + 1, num_batches, status, batch_loss);
            }
            
            // 训练完成后换行
            std::cout << std::endl;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = end_time - start_time;
            
            epoch_loss /= num_batches;
            
            // 计算训练准确率
            Tensor train_pred = model.predict(train_images);
            float train_accuracy = calculate_accuracy(train_pred, train_labels);
            
            // 计算测试准确率
            Tensor test_pred = model.predict(test_images);
            float test_accuracy = calculate_accuracy(test_pred, test_labels);
            
            Ctorch_Error::info(ErrorPlatform::kAutoDiff, "Epoch: " + std::to_string(epoch+1) + ", Average Loss: " + std::to_string(epoch_loss) + ", Train Acc: " + std::to_string(train_accuracy * 100) + "%, Test Acc: " + std::to_string(test_accuracy * 100) + "%, Time: " + std::to_string(elapsed.count()) + "s");
        }
        
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "训练完成");
        
        // 评估模型
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "评估模型...");
        
        // 计算测试集准确率
        Tensor test_pred = model.predict(test_images);
        float test_accuracy = calculate_accuracy(test_pred, test_labels);
        
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "测试集准确率: " + std::to_string(test_accuracy * 100) + "%");
        
        // 计算训练集准确率
        Tensor train_pred = model.predict(train_images);
        float train_accuracy = calculate_accuracy(train_pred, train_labels);
        
        Ctorch_Error::info(ErrorPlatform::kAutoDiff, "训练集准确率: " + std::to_string(train_accuracy * 100) + "%");
        
    } catch (const std::exception& e) {
        Ctorch_Error::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "错误: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}
