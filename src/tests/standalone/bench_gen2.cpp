/**
 * @file bench_gen2.cpp
 * @brief Gen 2 分布式系统性能基准测试
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/05
 *
 * 测量指标：
 * 1. CDTF 序列化/反序列化吞吐量（不同张量大小）
 * 2. 不同压缩策略的压缩比和精度损失
 * 3. Transport 消息延迟（不同消息大小）
 * 4. Transport 吞吐量（批量消息）
 * 5. 端到端梯度传输延迟（序列化+传输+反序列化）
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>

#include "Tensor.h"
#include "Distributed/Transport.h"
#include "Distributed/CDTF.h"
#include "Distributed/EntropyAwareCompressor.h"
#include "CtorchScheduler.h"

using namespace ct::distributed;

// ======================= 计时辅助 =======================

class Timer {
public:
    void start() { _start = std::chrono::steady_clock::now(); }
    double elapsedMs() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - _start).count();
    }
    double elapsedUs() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(end - _start).count();
    }
private:
    std::chrono::steady_clock::time_point _start;
};

// ======================= 辅助函数 =======================

static Tensor makeRandomTensor(size_t numel, float scale = 1.0f) {
    Tensor t(ShapeTag{}, {numel}, DType::kFloat, DeviceType::kCPU, false);
    float* d = t.data_write<float>();
    for (size_t i = 0; i < numel; ++i) {
        d[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }
    return t;
}

static std::string formatSize(size_t bytes) {
    if (bytes < 1024) return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes / (1024 * 1024)) + " MB";
}

// ======================= 基准 1: CDTF 序列化吞吐量 =======================

void bench_cdtf_serialization() {
    std::cout << "\n--- Benchmark 1: CDTF Serialization/Deserialization Throughput ---\n";

    // 不同张量大小
    std::vector<size_t> sizes = {
        1024,           // 4 KB
        1024 * 1024,    // 4 MB
        4 * 1024 * 1024 // 16 MB
    };

    for (auto numel : sizes) {
        Tensor t = makeRandomTensor(numel, 1.0f);

        // 序列化
        {
            constexpr int kIter = 100;
            Timer timer;
            timer.start();
            for (int i = 0; i < kIter; ++i) {
                auto data = CDTF::serialize(t, CDTF_FLAG_NONE);
                (void)data;
            }
            double ms = timer.elapsedMs();
            double total_bytes = static_cast<double>(numel) * sizeof(float) * kIter;
            double throughput = total_bytes / (ms / 1000.0) / (1024.0 * 1024.0);

            std::cout << "  Serialize " << formatSize(numel * sizeof(float))
                      << ": " << std::fixed << std::setprecision(2)
                      << (ms / kIter) << " ms/op, "
                      << throughput << " MB/s\n";
        }

        // 反序列化
        {
            auto data = CDTF::serialize(t, CDTF_FLAG_NONE);
            constexpr int kIter = 100;
            Timer timer;
            timer.start();
            for (int i = 0; i < kIter; ++i) {
                Tensor deserialized = CDTF::deserialize(data);
                (void)deserialized;
            }
            double ms = timer.elapsedMs();
            double total_bytes = static_cast<double>(data.size()) * kIter;
            double throughput = total_bytes / (ms / 1000.0) / (1024.0 * 1024.0);

            std::cout << "  Deserialize " << formatSize(numel * sizeof(float))
                      << ": " << std::fixed << std::setprecision(2)
                      << (ms / kIter) << " ms/op, "
                      << throughput << " MB/s\n";
        }
    }
}

// ======================= 基准 1b: CDTF 量化序列化吞吐量 =======================

void bench_cdtf_quantized_throughput() {
    std::cout << "\n--- Benchmark 1b: CDTF Quantized Serialization Throughput ---\n";

    constexpr size_t kNumel = 1024 * 1024; // 4 MB 张量
    constexpr int kIter = 50;

    // 高熵和低熵两种数据
    struct TestCase {
        const char* name;
        Tensor tensor;
    };
    TestCase cases[] = {
        {"High-Entropy", makeRandomTensor(kNumel, 1.0f)},
        {"Low-Entropy",  makeRandomTensor(kNumel, 0.001f)},
    };

    // 压缩策略
    struct Strategy {
        const char* name;
        uint16_t flags;
    };
    Strategy strategies[] = {
        {"Uncompressed", CDTF_FLAG_NONE},
        {"Float16",      CDTF_FLAG_QUANTIZE_16},
        {"Int8",         CDTF_FLAG_QUANTIZE_8},
    };

    std::cout << "  " << std::left << std::setw(16) << "Strategy"
              << std::setw(14) << "Type"
              << std::setw(16) << "Serialize"
              << std::setw(16) << "Deserialize"
              << std::setw(14) << "Ratio"
              << std::setw(14) << "Max Error"
              << "\n";
    std::cout << "  " << std::string(90, '-') << "\n";

    for (auto& s : strategies) {
        for (auto& c : cases) {
            // 预序列化一次获取压缩数据，用于反序列化基准
            std::vector<uint8_t> pre_serialized = CDTF::serialize(c.tensor, s.flags);
            size_t raw_size = c.tensor.numel() * sizeof(float);
            double ratio = static_cast<double>(pre_serialized.size()) / raw_size;

            // 验证精度
            Tensor deserialized = CDTF::deserialize(pre_serialized);
            const float* orig = c.tensor.data_read<float>();
            const float* des = deserialized.data_read<float>();
            float max_err = 0.0f;
            for (size_t i = 0; i < c.tensor.numel(); ++i) {
                float err = std::abs(orig[i] - des[i]);
                if (err > max_err) max_err = err;
            }

            // 序列化吞吐量
            double serialize_ms;
            {
                Timer timer;
                timer.start();
                for (int i = 0; i < kIter; ++i) {
                    auto data = CDTF::serialize(c.tensor, s.flags);
                    (void)data;
                }
                serialize_ms = timer.elapsedMs() / kIter;
            }

            // 反序列化吞吐量
            double deserialize_ms;
            {
                Timer timer;
                timer.start();
                for (int i = 0; i < kIter; ++i) {
                    Tensor t = CDTF::deserialize(pre_serialized);
                    (void)t;
                }
                deserialize_ms = timer.elapsedMs() / kIter;
            }

            double serialize_bw = (raw_size / (1024.0 * 1024.0)) / (serialize_ms / 1000.0);
            double deserialize_bw = (raw_size / (1024.0 * 1024.0)) / (deserialize_ms / 1000.0);

            std::cout << "  " << std::left << std::setw(16) << s.name
                      << std::setw(14) << c.name
                      << std::setw(12) << std::fixed << std::setprecision(1) << serialize_bw << " MB/s"
                      << std::setw(12) << std::fixed << std::setprecision(1) << deserialize_bw << " MB/s"
                      << std::setw(14) << std::fixed << std::setprecision(3) << ratio
                      << std::setw(14) << std::scientific << std::setprecision(2) << max_err
                      << "\n";
        }
    }
}

// ======================= 基准 2: 压缩策略对比 =======================

void bench_compression_strategies() {
    std::cout << "\n--- Benchmark 2: Compression Strategy Comparison ---\n";

    // 低熵梯度（接近零）
    Tensor low_entropy = makeRandomTensor(1024 * 1024, 0.001f);
    // 高熵梯度（均匀分布）
    Tensor high_entropy = makeRandomTensor(1024 * 1024, 1.0f);

    // 获取原始数据用于精度比较
    const float* low_orig = low_entropy.data_read<float>();
    const float* high_orig = high_entropy.data_read<float>();

    struct Strategy {
        const char* name;
        uint16_t flags;
    };

    Strategy strategies[] = {
        {"Uncompressed", CDTF_FLAG_NONE},
        {"Float16",      CDTF_FLAG_QUANTIZE_16},
        {"Int8+Entropy", CDTF_FLAG_QUANTIZE_8 | CDTF_FLAG_COMPRESSED},
    };

    std::cout << "  " << std::left << std::setw(20) << "Strategy"
              << std::setw(12) << "Type"
              << std::setw(14) << "Raw Size"
              << std::setw(14) << "Compressed"
              << std::setw(12) << "Ratio"
              << std::setw(16) << "Max Error"
              << "\n";
    std::cout << "  " << std::string(88, '-') << "\n";

    for (auto& s : strategies) {
        for (int is_low = 0; is_low <= 1; ++is_low) {
            const Tensor& t = is_low ? low_entropy : high_entropy;
            const char* type = is_low ? "Low-Entropy" : "High-Entropy";

            auto data = CDTF::serialize(t, s.flags);
            Tensor result = CDTF::deserialize(data);

            const float* res_data = result.data_read<float>();
            const float* orig = is_low ? low_orig : high_orig;

            float max_err = 0.0f;
            for (size_t i = 0; i < t.numel(); ++i) {
                float err = std::abs(orig[i] - res_data[i]);
                if (err > max_err) max_err = err;
            }

            size_t raw_size = t.numel() * sizeof(float);
            double ratio = static_cast<double>(data.size()) / raw_size;

            std::cout << "  " << std::left << std::setw(20) << s.name
                      << std::setw(12) << type
                      << std::setw(14) << formatSize(raw_size)
                      << std::setw(14) << formatSize(data.size())
                      << std::setw(12) << std::fixed << std::setprecision(3) << ratio
                      << std::setw(16) << std::scientific << std::setprecision(2) << max_err
                      << "\n";
        }
    }
}

// ======================= 基准 3: Transport 延迟 =======================

void bench_transport_latency() {
    std::cout << "\n--- Benchmark 3: Transport Message Latency ---\n";

    Transport::Config cfg0{0, 0};
    Transport::Config cfg1{1, 0};
    Transport t0(cfg0);
    Transport t1(cfg1);

    if (!t0.start() || !t1.start()) {
        std::cerr << "  FAILED: Transport start\n";
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (!t1.connectToPeer(0, "127.0.0.1", t0.localPort())) {
        std::cerr << "  FAILED: Transport connect\n";
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::atomic<int> received{0};
    int expected = 0;
    t0.setReceiveCallback([&](uint32_t, const std::vector<uint8_t>&) {
        received++;
    });

    // 不同消息大小
    std::vector<size_t> msg_sizes = {64, 1024, 16384, 65536, 262144, 1048576};
    constexpr int kIter = 100;

    std::cout << "  " << std::left << std::setw(14) << "Msg Size"
              << std::setw(14) << "Avg Latency"
              << std::setw(14) << "Min Latency"
              << std::setw(14) << "Max Latency"
              << "\n";
    std::cout << "  " << std::string(56, '-') << "\n";

    for (auto msg_size : msg_sizes) {
        std::vector<uint8_t> msg(msg_size, 0x42);
        expected += kIter;

        double min_us = 1e9, max_us = 0, sum_us = 0;

        for (int i = 0; i < kIter; ++i) {
            int prev = received.load();
            Timer timer;
            timer.start();
            t1.send(0, msg);

            // 等待 ACK
            while (received.load() == prev) {
                std::this_thread::yield();
            }
            double us = timer.elapsedUs();
            sum_us += us;
            if (us < min_us) min_us = us;
            if (us > max_us) max_us = us;
        }

        std::cout << "  " << std::left << std::setw(14) << formatSize(msg_size)
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << (sum_us / kIter) << " us"
                  << std::setw(12) << min_us << " us"
                  << std::setw(12) << max_us << " us"
                  << "\n";
    }

    t0.stop();
    t1.stop();
}

// ======================= 基准 4: Transport 吞吐量 =======================

void bench_transport_throughput() {
    std::cout << "\n--- Benchmark 4: Transport Throughput ---\n";

    Transport::Config cfg0{0, 0};
    Transport::Config cfg1{1, 0};
    Transport t0(cfg0);
    Transport t1(cfg1);

    if (!t0.start() || !t1.start()) {
        std::cerr << "  FAILED: Transport start\n";
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (!t1.connectToPeer(0, "127.0.0.1", t0.localPort())) {
        std::cerr << "  FAILED: Transport connect\n";
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::atomic<size_t> total_bytes{0};
    std::atomic<bool> done{false};

    t0.setReceiveCallback([&](uint32_t, const std::vector<uint8_t>& data) {
        total_bytes += data.size();
    });

    // 发送大块数据测量吞吐量
    constexpr size_t kChunkSize = 65536;  // 64 KB 块
    constexpr size_t kTotalSize = 32 * 1024 * 1024; // 32 MB 总数据
    constexpr int kChunks = kTotalSize / kChunkSize;

    std::vector<uint8_t> chunk(kChunkSize, 0x42);

    Timer timer;
    timer.start();
    for (int i = 0; i < kChunks; ++i) {
        t1.send(0, chunk);
    }

    // 等待接收完成
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    double ms = timer.elapsedMs();
    double throughput = static_cast<double>(kTotalSize) / (ms / 1000.0) / (1024.0 * 1024.0);

    std::cout << "  Sent " << formatSize(kTotalSize) << " in " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput << " MB/s\n";
    if (total_bytes.load() > 0) {
        std::cout << "  Received: " << formatSize(total_bytes.load()) << "\n";
    }

    t0.stop();
    t1.stop();
}

// ======================= 基准 5: 端到端梯度传输延迟 =======================

void bench_e2e_gradient_transmission() {
    std::cout << "\n--- Benchmark 5: End-to-End Gradient Transmission Latency ---\n";

    Transport::Config cfg0{0, 0};
    Transport::Config cfg1{1, 0};
    Transport t0(cfg0);
    Transport t1(cfg1);

    if (!t0.start() || !t1.start()) {
        std::cerr << "  FAILED: Transport start\n";
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (!t1.connectToPeer(0, "127.0.0.1", t0.localPort())) {
        std::cerr << "  FAILED: Transport connect\n";
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::atomic<int> received{0};
    std::vector<uint8_t> last_data;
    std::atomic<bool> data_ready{false};

    t0.setReceiveCallback([&](uint32_t, const std::vector<uint8_t>& data) {
        last_data = data;
        data_ready.store(true);
        received++;
    });

    std::vector<size_t> grad_sizes = {1024, 16384, 262144};
    constexpr int kIter = 50;

    std::cout << "  " << std::left << std::setw(14) << "Grad Size"
              << std::setw(14) << "Total Latency"
              << "\n";
    std::cout << "  " << std::string(28, '-') << "\n";

    for (auto numel : grad_sizes) {
        Tensor grad = makeRandomTensor(numel, 1.0f);

        double min_us = 1e9, max_us = 0, sum_us = 0;

        for (int i = 0; i < kIter; ++i) {
            data_ready.store(false);
            Timer timer;
            timer.start();

            // 序列化 + 传输 + 反序列化
            auto data = CDTF::serialize(grad, CDTF_FLAG_NONE);
            t1.send(0, data);

            // 等待接收
            while (!data_ready.load()) {
                std::this_thread::yield();
            }

            double us = timer.elapsedUs();
            sum_us += us;
            if (us < min_us) min_us = us;
            if (us > max_us) max_us = us;
        }

        std::cout << "  " << std::left << std::setw(14) << formatSize(numel * sizeof(float))
                  << std::fixed << std::setprecision(1)
                  << std::setw(14) << (sum_us / kIter) << " us"
                  << "\n";
    }

    t0.stop();
    t1.stop();
}

// ======================= 主函数 =======================

int main() {
    CtorchScheduler::getInstance();

    std::cout << "\n=============================================\n";
    std::cout << "  CTorch Gen 2 Performance Benchmarks\n";
    std::cout << "=============================================\n";

    bench_cdtf_serialization();
    bench_cdtf_quantized_throughput();
    bench_compression_strategies();
    bench_transport_latency();
    bench_transport_throughput();
    bench_e2e_gradient_transmission();

    std::cout << "\n=============================================\n";
    std::cout << "  Benchmarks Complete\n";
    std::cout << "=============================================\n\n";

    return 0;
}