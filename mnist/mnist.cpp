#include "mnist_loader.h"
#include "AutoDiff.h"
#include "Ctorch_Error.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// 两隐藏层 MLP: 784 -> 256(ReLU) -> 128(ReLU) -> 10
class NeuralNetwork {
private:
    Tensor W1, b1, W2, b2, W3, b3;
    float learning_rate;
    
    static void xavier_init(Tensor& W, size_t fan_in) {
        float std = std::sqrt(1.0f / fan_in);
        for (size_t i = 0; i < W.numel(); ++i) {
            float r = (2.0f * rand() / static_cast<float>(RAND_MAX)) - 1.0f;
            W.data<float>()[i] = r * std;
        }
    }
    
public:
    NeuralNetwork(int input_size, int hidden1, int hidden2, int output_size, float lr) : 
        learning_rate(lr) {
        W1 = Tensor(ShapeTag{}, {static_cast<size_t>(input_size), static_cast<size_t>(hidden1)}, DType::kFloat, DeviceType::kCPU);
        b1 = Tensor(ShapeTag{}, {static_cast<size_t>(hidden1)}, DType::kFloat, DeviceType::kCPU);
        W2 = Tensor(ShapeTag{}, {static_cast<size_t>(hidden1), static_cast<size_t>(hidden2)}, DType::kFloat, DeviceType::kCPU);
        b2 = Tensor(ShapeTag{}, {static_cast<size_t>(hidden2)}, DType::kFloat, DeviceType::kCPU);
        W3 = Tensor(ShapeTag{}, {static_cast<size_t>(hidden2), static_cast<size_t>(output_size)}, DType::kFloat, DeviceType::kCPU);
        b3 = Tensor(ShapeTag{}, {static_cast<size_t>(output_size)}, DType::kFloat, DeviceType::kCPU);
        
        xavier_init(W1, input_size);
        xavier_init(W2, hidden1);
        xavier_init(W3, hidden2);
        b1.zero(); b2.zero(); b3.zero();
    }
    
    Tensor forward(const Tensor& x) {
        Tensor h1 = (x.matmul(W1) + b1).relu();
        Tensor h2 = (h1.matmul(W2) + b2).relu();
        return h2.matmul(W3) + b3;
    }
    
    float train_step(const Tensor& x, const Tensor& y) {
        if (AutoDiffContext::current()) {
            AutoDiffContext::current()->make_leaf(W1, true);
            AutoDiffContext::current()->make_leaf(b1, true);
            AutoDiffContext::current()->make_leaf(W2, true);
            AutoDiffContext::current()->make_leaf(b2, true);
            AutoDiffContext::current()->make_leaf(W3, true);
            AutoDiffContext::current()->make_leaf(b3, true);
        }
        W1.requires_grad(true); b1.requires_grad(true);
        W2.requires_grad(true); b2.requires_grad(true);
        W3.requires_grad(true); b3.requires_grad(true);
        
        if (AutoDiffContext::current()) {
            AutoDiffContext::current()->zero_grad(W1);
            AutoDiffContext::current()->zero_grad(b1);
            AutoDiffContext::current()->zero_grad(W2);
            AutoDiffContext::current()->zero_grad(b2);
            AutoDiffContext::current()->zero_grad(W3);
            AutoDiffContext::current()->zero_grad(b3);
        }
        
        Tensor logits = forward(x);
        Tensor y_one_hot = Tensor(ShapeTag{}, {static_cast<size_t>(y.numel()), 10}, DType::kFloat, DeviceType::kCPU);
        for (size_t i = 0; i < y.numel(); ++i) {
            int lab = static_cast<int>(y.data<float>()[i]);
            for (int j = 0; j < 10; ++j)
                y_one_hot.data<float>()[i * 10 + j] = (j == lab) ? 1.0f : 0.0f;
        }
        Tensor loss = logits.cross_entropy(y_one_hot);
        float loss_value = loss.item<float>();
        backward(loss);
        update_parameters();
        return loss_value;
    }
    
    void update_parameters() {
        float lr = learning_rate;
        auto sgd_step = [this, lr](Tensor& param) {
            Tensor g = grad(param);
            float* p = param.data<float>();
            const float* gp = g.data<float>();
            for (size_t i = 0; i < param.numel(); ++i)
                p[i] -= gp[i] * lr;
        };
        sgd_step(W1); sgd_step(b1);
        sgd_step(W2); sgd_step(b2);
        sgd_step(W3); sgd_step(b3);
    }
    
    // 预测（这里不需要计算梯度，手动实现按行 softmax，确保每个样本的10类概率和为1）
    Tensor predict(const Tensor& x) {
        Tensor logits = forward(x);  // [batch_size, 10]
        std::vector<size_t> shape = logits.sizes();
        if (shape.size() != 2 || shape[1] != 10) {
            // 防御性检查：如果形状不符合预期，直接返回 logits
            return logits;
        }
        size_t batch_size = shape[0];
        size_t num_classes = shape[1];
        
        Tensor y_pred(ShapeTag{}, shape, logits.dtype(), logits.device());
        const float* in = logits.data<float>();
        float* out = y_pred.data<float>();
        
        for (size_t i = 0; i < batch_size; ++i) {
            // 数值稳定的 softmax：先减去该样本的最大值
            float max_val = in[i * num_classes];
            for (size_t j = 1; j < num_classes; ++j) {
                float v = in[i * num_classes + j];
                if (v > max_val) max_val = v;
            }
            
            float exp_sum = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                float e = std::exp(in[i * num_classes + j] - max_val);
                out[i * num_classes + j] = e;
                exp_sum += e;
            }
            // 归一化
            for (size_t j = 0; j < num_classes; ++j) {
                out[i * num_classes + j] /= exp_sum;
            }
        }
        
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
        // MINIUM=少输出, FULL=显示TRACE(含阶段3、W2梯度验证)
        Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
        
        // 设置随机种子
        srand(42);
        
        // 加载MNIST数据

        MNISTLoader loader(".");
        Tensor train_images, train_labels;
        Tensor test_images, test_labels;
        loader.load_training_data(train_images, train_labels);
        loader.load_test_data(test_images, test_labels);
        
        std::cout << "MNIST 加载完成 | 训练:" << train_images.shape()[0] << " 测试:" << test_images.shape()[0] << std::endl;
        
        int input_size = 784;
        int hidden1 = 128, hidden2 = 64;  // 简化：256->128, 128->64
        int output_size = 10;
        float learning_rate = 0.01f;  // 提高学习率加速收敛
        
        NeuralNetwork model(input_size, hidden1, hidden2, output_size, learning_rate);
        
        int epochs = 5;  // 减少epochs
        int batch_size = 128;  // 增大batch size减少batch数量
        int num_batches = static_cast<int>(train_images.shape()[0]) / batch_size;
        
        std::cout << "网络: 784->" << hidden1 << "->" << hidden2 << "->10 | Epochs:" << epochs << " | Batch:" << batch_size << " | lr:" << learning_rate << std::endl;
        
        // 创建一个AutoDiff上下文用于整个训练过程
        AutoDiff autodiff;
        AutoDiffContext::Guard guard(&autodiff);
        
        // 完整训练：遍历所有 epoch 和 batch
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            for (int batch = 0; batch < num_batches; ++batch) {
                Tensor batch_images, batch_labels;
                get_batch(train_images, train_labels, batch_size, batch, batch_images, batch_labels);
                batch_images.requires_grad(false);
                batch_labels.requires_grad(false);
                
                float batch_loss = model.train_step(batch_images, batch_labels);
                epoch_loss += batch_loss;
                
                if (batch % 50 == 0 || batch == num_batches - 1) {
                    show_progress(epoch * num_batches + batch + 1, epochs * num_batches, 
                                 "E" + std::to_string(epoch + 1) + "/" + std::to_string(epochs), 
                                 batch_loss);
                }
            }
            epoch_loss /= static_cast<float>(num_batches);
            std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << " Loss: " 
                      << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        }
        
        Tensor test_pred = model.predict(test_images);
        float test_acc = calculate_accuracy(test_pred, test_labels);
        Tensor train_pred = model.predict(train_images);
        float train_acc = calculate_accuracy(train_pred, train_labels);
        
        std::cout << "\n>>> 训练集准确率: " << std::fixed << std::setprecision(2) << (train_acc * 100) << "%" << std::endl;
        std::cout << ">>> 测试集准确率: " << std::fixed << std::setprecision(2) << (test_acc * 100) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        Ctorch_Error::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "错误: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}
