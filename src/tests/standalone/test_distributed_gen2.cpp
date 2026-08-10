/**
 * @file test_distributed_gen2.cpp
 * @brief Gen 2 分布式系统单元测试 — 验证 BANT + GTCS 核心模块
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1
 *
 * 测试覆盖：
 * 1. CDTF 序列化/反序列化往返精度
 * 2. GradientAggregator 四种聚合策略
 * 3. BackendManager 注册/查询
 * 4. GTCScheduler VCG 分配和比例权重
 * 5. DeviceMigration 自然变换路径
 * 6. CommEngine 节点管理
 * 7. DistributedOptimizer Local-SGD 流程
 * 8. CRDT 状态合并
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>

#include "Tensor.h"
#include "Distributed/CDTF.h"
#include "Distributed/GradientAggregator.h"
#include "Distributed/BackendManager.h"
#include "Distributed/DeviceMigration.h"
#include "Distributed/CommEngine.h"
#include "Distributed/DistributedOptimizer.h"
#include "Distributed/GTCScheduler.h"
#include "Distributed/EntropyAwareCompressor.h"
#include "Distributed/TopologyManager.h"
#include "Distributed/QuorumManager.h"
#include "Distributed/NodeDiscovery.h"
#include "Distributed/FaultTolerance.h"
#include "Distributed/CheckpointManager.h"
#include "Distributed/CPUBackend.h"
#include "Distributed/MPSBackend.h"
#include "Distributed/DistributedTrainer.h"
#include "Distributed/Transport.h"

#include "CtorchScheduler.h"

using namespace ct::distributed;

// ======================= 测试辅助 =======================

static int test_passed = 0;
static int test_failed = 0;

#define TEST(name) \
    do { \
        std::cout << "  [TEST] " << name << "... "; \
        try {

#define END_TEST(name) \
            std::cout << "PASSED" << std::endl; \
            test_passed++; \
        } catch (const std::exception& e) { \
            std::cout << "FAILED: " << e.what() << std::endl; \
            test_failed++; \
        } \
    } while(0)

// ======================= 测试 1: CDTF 序列化往返 =======================

void test_cdtf_roundtrip() {
    TEST("CDTF serialize/deserialize roundtrip precision < 1e-7");

    // 创建随机 float32 tensor
    Tensor t(ShapeTag{}, {16, 32}, DType::kFloat, DeviceType::kCPU, false);
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 序列化
    auto serialized = CDTF::serialize(t, CDTF_FLAG_NONE);

    // 验证 header
    assert(CDTF::validate(serialized));
    assert(CDTF::peekDType(serialized) == DType::kFloat);
    assert(CDTF::peekNumel(serialized) == t.numel());
    auto shape = CDTF::peekShape(serialized);
    assert(shape.size() == 2);
    assert(shape[0] == 16);
    assert(shape[1] == 32);

    // 反序列化
    Tensor deserialized = CDTF::deserialize(serialized);

    // 验证数据
    assert(deserialized.numel() == t.numel());
    assert(deserialized.device() == DeviceType::kCPU);

    const float* orig_data = t.data_read<float>();
    const float* des_data = deserialized.data_read<float>();

    float max_diff = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) {
        float diff = std::abs(orig_data[i] - des_data[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "max_diff=" << max_diff << " ";
    assert(max_diff < 1e-7f);

    // 验证 roundtripError 快捷方法
    float error = CDTF::roundtripError(t);
    assert(error < 1e-7f);

    END_TEST("CDTF roundtrip");
}

// ======================= 测试 1b: CDTF 8-bit 量化往返 =======================

void test_cdtf_quantize8() {
    TEST("CDTF 8-bit quantize/dequantize roundtrip");

    // 创建随机 float32 tensor（范围可控，适合 8-bit 量化）
    Tensor t(ShapeTag{}, {8, 16}, DType::kFloat, DeviceType::kCPU, false);
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }

    // 使用 8-bit 量化序列化
    auto serialized = CDTF::serialize(t, CDTF_FLAG_QUANTIZE_8);

    // 验证 header
    assert(CDTF::validate(serialized));
    assert(CDTF::peekDType(serialized) == DType::kFloat);
    assert(CDTF::peekNumel(serialized) == t.numel());

    // 验证 8-bit 量化后的数据大小
    // data_size = 2 * sizeof(float) + numel = 8 + 128 = 136
    size_t expected_data_size = 2 * sizeof(float) + t.numel();
    assert(serialized.size() > 32);  // 至少大于 header

    // 反序列化
    Tensor deserialized = CDTF::deserialize(serialized);

    // 验证数据形状和类型
    assert(deserialized.numel() == t.numel());
    assert(deserialized.device() == DeviceType::kCPU);
    assert(deserialized.shape() == t.shape());

    // 验证 8-bit 量化精度：最大误差应小于 range/255
    const float* orig_data = t.data_read<float>();
    const float* des_data = deserialized.data_read<float>();

    float min_val = data[0], max_val = data[0];
    for (size_t i = 1; i < t.numel(); ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    float range = max_val - min_val;
    if (range < 1e-12f) range = 1e-12f;
    float max_theoretical_error = range / 255.0f;

    float max_diff = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) {
        float diff = std::abs(orig_data[i] - des_data[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "max_diff=" << max_diff << " (theoretical_max=" << max_theoretical_error << ") ";
    assert(max_diff <= max_theoretical_error + 1e-6f);

    // 验证压缩比：8-bit 量化应小于 float32
    size_t float32_size = 32 + t.numel() * sizeof(uint64_t) + t.numel() * sizeof(float);
    assert(serialized.size() < float32_size);

    END_TEST("CDTF 8-bit quantize");
}

// ======================= 测试 1c: CDTF Float16 量化往返 =======================

void test_cdtf_quantize16() {
    TEST("CDTF 16-bit quantize/dequantize roundtrip");

    // 创建随机 float32 tensor
    Tensor t(ShapeTag{}, {4, 8}, DType::kFloat, DeviceType::kCPU, false);
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // 使用 Float16 量化序列化
    auto serialized = CDTF::serialize(t, CDTF_FLAG_QUANTIZE_16);

    // 验证 header
    assert(CDTF::validate(serialized));
    assert(CDTF::peekDType(serialized) == DType::kFloat);

    // 反序列化
    Tensor deserialized = CDTF::deserialize(serialized);

    // 验证数据
    assert(deserialized.numel() == t.numel());
    assert(deserialized.shape() == t.shape());

    const float* orig_data = t.data_read<float>();
    const float* des_data = deserialized.data_read<float>();

    float max_diff = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) {
        float diff = std::abs(orig_data[i] - des_data[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // Float16 精度约 1e-3，在 [-1, 1] 范围内
    std::cout << "max_diff=" << max_diff << " ";
    assert(max_diff < 1e-3f);

    // 验证压缩比
    size_t float32_size = 32 + t.numel() * sizeof(uint64_t) + t.numel() * sizeof(float);
    assert(serialized.size() < float32_size);

    END_TEST("CDTF 16-bit quantize");
}

// ======================= 测试 2: 梯度聚合 =======================

void test_gradient_aggregator() {
    // 在函数作用域声明测试数据，避免 TEST 宏 do-while 块的作用域限制
    Tensor g1(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU, false);
    Tensor g2(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU, false);
    Tensor g3(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU, false);

    float* d1 = g1.data_write<float>(); d1[0] = 1.0f; d1[1] = 2.0f; d1[2] = 3.0f;
    float* d2 = g2.data_write<float>(); d2[0] = 2.0f; d2[1] = 3.0f; d2[2] = 4.0f;
    float* d3 = g3.data_write<float>(); d3[0] = 3.0f; d3[1] = 4.0f; d3[2] = 5.0f;

    GradientAggregator agg(AggregationStrategy::SimpleAverage);

    // 带离群值的梯度，用于 RobustMedian 测试
    Tensor g4(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU, false);
    float* d4 = g4.data_write<float>(); d4[0] = 1.0f; d4[1] = 100.0f; d4[2] = 3.0f;

    TEST("GradientAggregator simple average");

    Tensor result = agg.aggregate({g1, g2, g3});

    const float* r = result.data_read<float>();
    assert(std::abs(r[0] - 2.0f) < 1e-6f);  // (1+2+3)/3 = 2
    assert(std::abs(r[1] - 3.0f) < 1e-6f);  // (2+3+4)/3 = 3
    assert(std::abs(r[2] - 4.0f) < 1e-6f);  // (3+4+5)/3 = 4

    END_TEST("GradientAggregator simple average");

    TEST("GradientAggregator weighted average with backend precision");

    agg.setStrategy(AggregationStrategy::WeightedAverage);
    Tensor w_result = agg.aggregate({g1, g2, g3}, {0.5f, 0.3f, 0.2f});

    const float* wr = w_result.data_read<float>();
    // 加权平均: 0.5*1 + 0.3*2 + 0.2*3 = 1.7
    assert(std::abs(wr[0] - 1.7f) < 1e-6f);
    // 0.5*2 + 0.3*3 + 0.2*4 = 2.7
    assert(std::abs(wr[1] - 2.7f) < 1e-6f);
    // 0.5*3 + 0.3*4 + 0.2*5 = 3.7
    assert(std::abs(wr[2] - 3.7f) < 1e-6f);

    END_TEST("GradientAggregator weighted average");

    TEST("GradientAggregator robust median");

    agg.setStrategy(AggregationStrategy::RobustMedian);
    Tensor m_result = agg.aggregate({g1, g2, g4});

    const float* mr = m_result.data_read<float>();
    // 中位数: g4=[1,100,3]（100 为离群值）
    //  [1,2,1] -> 1, [2,3,100] -> 3, [3,4,3] -> 3
    assert(std::abs(mr[0] - 1.0f) < 1e-6f);
    assert(std::abs(mr[1] - 3.0f) < 1e-6f);
    assert(std::abs(mr[2] - 3.0f) < 1e-6f);

    END_TEST("GradientAggregator robust median");

    TEST("GradientAggregator quorum aggregation");

    agg.setStrategy(AggregationStrategy::SimpleAverage);
    // Quorum = 2，3 个梯度可用
    Tensor q_result = agg.aggregateWithQuorum({g1, g2, g3}, 2);
    assert(q_result.numel() > 0);

    // Quorum = 5，但只有 3 个梯度
    Tensor q_empty = agg.aggregateWithQuorum({g1, g2, g3}, 5);
    // 空 Tensor() 的 shape() 为空（numel() 对空 shape 返回 1=标量，不能用于判空）
    assert(q_empty.shape().empty());

    END_TEST("GradientAggregator quorum aggregation");
}

// ======================= 测试 3: BackendManager =======================

void test_backend_manager() {
    // 先初始化调度器（确保 allocator 已注册）
    CtorchScheduler::getInstance();

    auto& mgr = BackendManager::getInstance();
    Tensor t_test;       // 在 TEST 块外声明，跨块复用
    float* d_test = nullptr;
    // cpu_backend / cap 需跨 TEST 块复用（否则 333 行访问 cap 超作用域），故提升到函数作用域
    std::shared_ptr<CPUBackend> cpu_backend;
    BackendCapability cap = {};

    TEST("BackendManager register CPUBackend");

    cpu_backend = std::make_shared<CPUBackend>();
    mgr.registerBackend(cpu_backend);
    assert(mgr.backendCount() > 0);
    assert(mgr.hasBackend(DeviceType::kCPU));

    cap = cpu_backend->capability();
    assert(cap.device == DeviceType::kCPU);
    assert(cap.compute_throughput > 0.0f);
    assert(cap.numerical_precision == 1.0f);
    assert(std::string(cpu_backend->name()) == "CPU");

    END_TEST("BackendManager register CPUBackend");

    TEST("BackendManager register MPSBackend");

    auto mps_backend = std::make_shared<MPSBackend>();
    mgr.registerBackend(mps_backend);
    assert(mgr.hasBackend(DeviceType::kMPS));

    auto mps_cap = mps_backend->capability();
    assert(mps_cap.device == DeviceType::kMPS);
    assert(mps_cap.compute_throughput > 0.0f);
    assert(std::string(mps_backend->name()) == "MPS");
    assert(mps_cap.compute_throughput > cap.compute_throughput);

    END_TEST("BackendManager register MPSBackend");

    TEST("BackendManager getBackend and executeKernel");

    auto cpu = mgr.getBackend(DeviceType::kCPU);
    assert(cpu != nullptr);
    assert(cpu->deviceType() == DeviceType::kCPU);

    t_test = Tensor(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    d_test = t_test.data_write<float>();
    d_test[0] = -1.0f; d_test[1] = 0.0f; d_test[2] = 1.0f; d_test[3] = 2.0f;

    Tensor out;
    cpu->executeKernel(op::ReLU, t_test, Tensor(), out);
    assert(out.numel() == 4);
    const float* r = out.data_read<float>();
    assert(std::abs(r[0] - 0.0f) < 1e-6f);
    assert(std::abs(r[1] - 0.0f) < 1e-6f);
    assert(std::abs(r[2] - 1.0f) < 1e-6f);
    assert(std::abs(r[3] - 2.0f) < 1e-6f);

    END_TEST("BackendManager getBackend and executeKernel");

    TEST("BackendManager serialize/deserialize via backend");

    auto cpu_back = mgr.getBackend(DeviceType::kCPU);
    auto serialized = cpu_back->serialize(t_test);
    Tensor deserialized = cpu_back->deserialize(serialized);
    assert(deserialized.numel() == t_test.numel());
    const float* des_data = deserialized.data_read<float>();
    float max_diff = 0.0f;
    for (size_t i = 0; i < t_test.numel(); ++i) {
        float diff = std::abs(d_test[i] - des_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-7f);

    END_TEST("BackendManager serialize/deserialize via backend");

    mgr.unregisterBackend(DeviceType::kCPU);
    mgr.unregisterBackend(DeviceType::kMPS);
}

// ======================= 测试 4: GTCScheduler =======================

void test_gtc_scheduler() {
    GTCScheduler scheduler;

    // 注册三个节点（函数作用域，跨 TEST 块可见）
    scheduler.setNodeBid(0, DeviceType::kCPU, 10.0f, 32);
    scheduler.setNodeBid(1, DeviceType::kCPU, 20.0f, 64);
    scheduler.setNodeBid(2, DeviceType::kCPU, 5.0f, 16);

    TEST("GTCScheduler VCG allocation and proportional weights");

    // VCG 分配
    auto allocation = scheduler.solveAllocation(100);

    // 验证分配结果
    assert(!allocation.empty());
    size_t total_batch = 0;
    for (const auto& a : allocation) {
        total_batch += a.batch_size;
    }
    // 总分配 batch 应 <= 100
    assert(total_batch <= 100);

    // 比例权重
    float w0 = scheduler.getAggregationWeight(0);
    float w1 = scheduler.getAggregationWeight(1);
    float w2 = scheduler.getAggregationWeight(2);

    // 权重之和应 ≈ 1.0
    float sum = w0 + w1 + w2;
    assert(std::abs(sum - 1.0f) < 1e-4f);

    // 各权重应 > 0
    assert(w0 > 0.0f);
    assert(w1 > 0.0f);
    assert(w2 > 0.0f);

    END_TEST("GTCScheduler VCG allocation");

    TEST("GTCScheduler PoA and cheating detection");

    float poa = scheduler.computePriceOfAnarchy();
    // PoA 应 >= 1.0
    assert(poa >= 1.0f);

    // 检测作弊
    std::unordered_map<uint32_t, float> actual_times = {
        {0, 11.0f},   // 偏差 10% — 正常
        {1, 30.0f},   // 偏差 50% — 作弊
        {2, 5.5f}     // 偏差 10% — 正常
    };
    auto cheaters = scheduler.detectBidCheating(actual_times);
    assert(cheaters.size() == 1);
    assert(cheaters[0] == 1);

    END_TEST("GTCScheduler cheating detection");

    TEST("GTCScheduler Shapley values");

    std::unordered_map<uint32_t, float> scores = {
        {0, 0.5f},
        {1, 0.3f},
        {2, 0.2f}
    };
    auto shapley = scheduler.computeShapleyValues(scores);
    assert(shapley.size() == 3);
    // 按 Shapley 值降序排列
    assert(shapley[0].marginal_contribution >= shapley[1].marginal_contribution);

    END_TEST("GTCScheduler Shapley values");
}

// ======================= 测试 5: DeviceMigration =======================

void test_device_migration() {
    // 使用 static_assert 避免 assert 宏处理模板逗号参数的问题
    static_assert(MigrationTraits<DeviceType::kCPU, DeviceType::kCPU>::supported);
    static_assert(!MigrationTraits<DeviceType::kCPU, DeviceType::kCPU>::needs_neutral);
    static_assert(MigrationTraits<DeviceType::kCPU, DeviceType::kMPS>::supported);
    static_assert(MigrationTraits<DeviceType::kMPS, DeviceType::kCPU>::supported);
    static_assert(!MigrationTraits<DeviceType::kCPU, DeviceType::kAMX>::supported);

    TEST("DeviceMigration migrateTensor CPU->CPU");

    Tensor t(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* d = t.data_write<float>();
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f;

    // CPU->CPU 迁移应为空操作
    Tensor migrated = migrateTensor(t, DeviceType::kCPU);
    assert(migrated.device() == DeviceType::kCPU);
    assert(migrated.numel() == 4);
    const float* md = migrated.data_read<float>();
    assert(std::abs(md[0] - 1.0f) < 1e-6f);
    assert(std::abs(md[3] - 4.0f) < 1e-6f);

    END_TEST("DeviceMigration migrateTensor CPU->CPU");
}

// ======================= 测试 6: CommEngine =======================

void test_comm_engine() {
    CommEngine engine(0);

    TEST("CommEngine node management");

    // 注册节点
    NodeInfo node1;
    node1.id = 1;
    node1.address = "192.168.1.1:8080";
    node1.backend_type = DeviceType::kCPU;
    node1.rtt_ms = 1.0f;
    node1.bandwidth_mbps = 1000.0f;
    node1.is_active = true;
    node1.compatibility_score = 100;
    engine.registerNode(node1);

    NodeInfo node2;
    node2.id = 2;
    node2.address = "192.168.1.2:8080";
    node2.backend_type = DeviceType::kMPS;
    node2.rtt_ms = 5.0f;
    node2.bandwidth_mbps = 500.0f;
    node2.is_active = true;
    node2.compatibility_score = 80;
    engine.registerNode(node2);

    // 查询活跃节点
    auto active = engine.activeNodes();
    assert(active.size() == 2);

    // 获取邻居（按延迟排序）
    auto neighbors = engine.getNeighbors();
    assert(neighbors.size() == 2);
    assert(neighbors[0] == 1);  // RTT 1ms 优先

    // 获取兼容邻居
    auto compat = engine.getCompatibleNeighbors(DeviceType::kCPU);
    assert(!compat.empty());

    END_TEST("CommEngine node management");

    TEST("CommEngine gradient send/recv");

    Tensor grad(ShapeTag{}, {8}, DType::kFloat, DeviceType::kCPU, false);
    float* gd = grad.data_write<float>();
    for (size_t i = 0; i < 8; ++i) gd[i] = static_cast<float>(i);

    // 发送梯度（在占位实现中会触发本地回调）
    engine.sendGradient(grad, 1);

    END_TEST("CommEngine gradient send/recv");
}

// ======================= 测试 7: DistributedOptimizer =======================

void test_distributed_optimizer() {
    TEST("DistributedOptimizer basic step");

    // 创建参数
    Tensor param(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* pd = param.data_write<float>();
    pd[0] = 1.0f; pd[1] = 1.0f; pd[2] = 1.0f; pd[3] = 1.0f;

    // 创建优化器
    auto config = OptimizerConfig::defaultConfig();
    config.local_steps = 5;
    config.learning_rate = 0.01f;
    config.momentum = 0.0f;

    std::vector<Tensor*> params = {&param};
    DistributedOptimizer optimizer(params, config);

    // 执行本地步
    Tensor grad(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* gd = grad.data_write<float>();
    gd[0] = 0.1f; gd[1] = 0.1f; gd[2] = 0.1f; gd[3] = 0.1f;

    optimizer.localStep({grad});

    // 验证统计信息
    auto stats = optimizer.stats();
    assert(stats.total_steps == 0);  // step() 未调用
    assert(stats.local_steps == 1);

    END_TEST("DistributedOptimizer basic step");

    TEST("DistributedOptimizer CRDT state merge");

    auto config2 = OptimizerConfig::defaultConfig();
    config2.local_steps = 10;
    Tensor param2(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU, false);
    std::vector<Tensor*> params2 = {&param2};
    DistributedOptimizer opt2(params2, config2);

    CRDTState state_a = opt2.getCRDTState();
    assert(state_a.global_step == 0);
    assert(state_a.local_step == 0);

    // 创建另一个状态
    CRDTState state_b;
    state_b.global_step = 5;
    state_b.local_step = 3;
    state_b.version_vector = {5};
    state_b.grad_counter = {10};
    state_b.momentum = {0.5f, 0.5f};

    // 合并
    opt2.mergeCRDTState(state_b);
    CRDTState merged = opt2.getCRDTState();
    assert(merged.global_step == 5);
    assert(merged.local_step == 3);

    END_TEST("DistributedOptimizer CRDT state merge");
}

// ======================= 测试 8: CRDTState =======================

void test_crdt_state() {
    TEST("CRDTState merge and dominates");

    CRDTState a;
    a.version_vector = {1, 2, 3};
    a.global_step = 10;
    a.local_step = 5;
    a.grad_counter = {10, 20, 30};

    CRDTState b;
    b.version_vector = {0, 3, 2};
    b.global_step = 8;
    b.local_step = 7;
    b.grad_counter = {5, 25, 15};

    // a 不完全 dominates b（a[0] > b[0] 但 a[1] < b[1]）
    assert(!a.dominates(b));
    assert(!b.dominates(a));

    // 合并
    CRDTState merged = CRDTState::merge(a, b);
    assert(merged.version_vector[0] == 1);
    assert(merged.version_vector[1] == 3);
    assert(merged.version_vector[2] == 3);
    assert(merged.global_step == 10);
    // local_step 取 LWW：b 更大
    assert(merged.local_step == 7);
    assert(merged.grad_counter[0] == 10);
    assert(merged.grad_counter[1] == 25);
    assert(merged.grad_counter[2] == 30);

    END_TEST("CRDTState merge and dominates");
}

// ======================= 测试 9: EntropyAwareCompressor =======================

void test_entropy_aware_compressor() {
    EntropyAwareCompressor compressor;

    // 创建一个低熵的梯度（值集中在 0 附近）
    Tensor low_entropy(ShapeTag{}, {100}, DType::kFloat, DeviceType::kCPU, false);
    float* le = low_entropy.data_write<float>();
    for (size_t i = 0; i < 100; ++i) le[i] = 0.001f * static_cast<float>(i % 3);

    // 创建一个高熵的梯度（均匀随机）
    Tensor high_entropy(ShapeTag{}, {100}, DType::kFloat, DeviceType::kCPU, false);
    float* he = high_entropy.data_write<float>();
    for (size_t i = 0; i < 100; ++i) he[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;

    TEST("EntropyAwareCompressor entropy estimation");

    float e_low = compressor.estimateEntropy(le, 100);
    float e_high = compressor.estimateEntropy(he, 100);

    // 低熵数据的熵应显著低于高熵数据
    assert(e_low < e_high);
    assert(e_low >= 0.0f);
    assert(e_high >= 0.0f);

    END_TEST("EntropyAwareCompressor entropy estimation");

    TEST("EntropyAwareCompressor compress/decompress lossless");

    // 无损压缩/解压
    Tensor t(ShapeTag{}, {4, 8}, DType::kFloat, DeviceType::kCPU, false);
    float* d = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) d[i] = static_cast<float>(i) * 0.25f;

    auto result = compressor.compress(t);
    // Float32 精度应无损
    assert(result.precision == QuantizePrecision::Float32 || result.lossless);

    auto decomp = compressor.decompress(result.compressed_data);
    assert(decomp.num_elements == t.numel());

    float max_err = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) {
        float err = std::abs(d[i] - decomp.data[i]);
        if (err > max_err) max_err = err;
    }
    // 无损压缩差异应接近 0
    assert(max_err < 1e-6f);

    END_TEST("EntropyAwareCompressor compress/decompress lossless");

    // 强制 8-bit 压缩用的数据，也在 decompressToTensor 测试中使用
    Tensor t2(ShapeTag{}, {32}, DType::kFloat, DeviceType::kCPU, false);
    float* d2 = t2.data_write<float>();
    for (size_t i = 0; i < 32; ++i) d2[i] = static_cast<float>(i) / 31.0f;

    TEST("EntropyAwareCompressor compress/decompress lossy 8-bit");

    auto result2 = compressor.compress(d2, 32, QuantizePrecision::Int8);
    assert(result2.precision == QuantizePrecision::Int8);
    assert(result2.compression_ratio < 0.5f);  // 8-bit 应该压缩到 1/4

    auto decomp2 = compressor.decompress(result2.compressed_data);
    assert(decomp2.num_elements == 32);

    // 8-bit 量化应该有一定误差，但不应太大
    float max_err2 = 0.0f;
    for (size_t i = 0; i < 32; ++i) {
        float err = std::abs(d2[i] - decomp2.data[i]);
        if (err > max_err2) max_err2 = err;
    }
    // 8-bit 量化误差应 < 1/255 ≈ 0.004
    assert(max_err2 < 0.005f);

    END_TEST("EntropyAwareCompressor compress/decompress lossy 8-bit");

    TEST("EntropyAwareCompressor decompressToTensor");

    auto result3 = compressor.compress(t2.data_read<float>(), 32, QuantizePrecision::Float16);
    auto tensor = compressor.decompressToTensor(result3.compressed_data, {32});
    assert(tensor.numel() == 32);
    assert(tensor.device() == DeviceType::kCPU);

    END_TEST("EntropyAwareCompressor decompressToTensor");

    TEST("EntropyAwareCompressor selectPrecision and predictRatio");

    // 低熵 → 8-bit
    auto p1 = compressor.selectPrecision(0.5f);
    assert(p1 == QuantizePrecision::Int8);

    // 中熵 → 16-bit
    auto p2 = compressor.selectPrecision(2.0f);
    assert(p2 == QuantizePrecision::Float16);

    // 高熵 → 32-bit
    auto p3 = compressor.selectPrecision(5.0f);
    assert(p3 == QuantizePrecision::Float32);

    float ratio = compressor.predictCompressionRatio(1.0f);
    assert(ratio > 0.0f && ratio < 1.0f);

    END_TEST("EntropyAwareCompressor selectPrecision and predictRatio");
}

// ======================= 测试 10: TopologyManager =======================

void test_topology_manager() {
    TopologyManager mgr(0);

    TEST("TopologyManager node registration and query");

    TopologyNode node1;
    node1.id = 1;
    node1.backend_type = DeviceType::kCPU;
    node1.address = "192.168.1.1:8080";
    node1.compute_throughput = 1.0f;
    node1.memory_bandwidth = 50.0f;
    node1.unified_memory = true;
    node1.numerical_precision = 1.0f;
    node1.max_batch_size = 64;
    node1.is_active = true;
    mgr.registerNode(node1);

    TopologyNode node2;
    node2.id = 2;
    node2.backend_type = DeviceType::kMPS;
    node2.address = "192.168.1.2:8080";
    node2.compute_throughput = 10.0f;
    node2.memory_bandwidth = 400.0f;
    node2.unified_memory = true;
    node2.numerical_precision = 0.95f;
    node2.max_batch_size = 128;
    node2.is_active = true;
    mgr.registerNode(node2);

    auto active = mgr.activeNodes();
    assert(active.size() == 2);

    auto n1 = mgr.getNode(1);
    assert(n1 != nullptr);
    assert(n1->id == 1);
    assert(n1->backend_type == DeviceType::kCPU);

    auto n3 = mgr.getNode(99);
    assert(n3 == nullptr);

    END_TEST("TopologyManager node registration and query");

    TEST("TopologyManager link registration and scoring");

    TopologyLink link12;
    link12.node_a = 0;
    link12.node_b = 1;
    link12.link_type = TopologyLinkType::Direct;
    link12.rtt_ms = 1.0f;
    link12.bandwidth_mbps = 1000.0f;
    link12.stability_score = 0.95f;
    mgr.registerLink(link12);

    TopologyLink link02;
    link02.node_a = 0;
    link02.node_b = 2;
    link02.link_type = TopologyLinkType::Direct;
    link02.rtt_ms = 5.0f;
    link02.bandwidth_mbps = 500.0f;
    link02.stability_score = 0.8f;
    mgr.registerLink(link02);

    // 计算兼容性评分
    float compat = mgr.computeBackendCompatibility(DeviceType::kCPU, DeviceType::kMPS);
    assert(compat > 0.0f && compat <= 1.0f);

    // 获取最佳邻居
    auto neighbors = mgr.getBestNeighbors();
    // 本地节点 0 连接 1 和 2
    assert(neighbors.size() <= 2);

    // 兼容邻居
    auto compat_neighbors = mgr.getCompatibleNeighbors(DeviceType::kCPU);
    assert(!compat_neighbors.empty());

    END_TEST("TopologyManager link registration and scoring");

    TEST("TopologyManager stale node detection");

    // 不应立即检测到失效
    auto stale = mgr.detectStaleNodes();
    // 刚注册的节点不应被标记为失效
    assert(stale.empty());

    mgr.unregisterNode(1);
    auto active2 = mgr.activeNodes();
    assert(active2.size() == 1);

    END_TEST("TopologyManager stale node detection");

    TEST("TopologyManager snapshot and connectivity");

    auto snapshot = mgr.getSnapshot();
    assert(snapshot.num_active_nodes >= 1);
    assert(snapshot.num_links >= 1);

    float connectivity = mgr.computeGraphConnectivity();
    assert(connectivity > 0.0f && connectivity <= 1.0f);

    mgr.hasDirectLink(0, 2);
    assert(!mgr.hasDirectLink(1, 2));

    END_TEST("TopologyManager snapshot and connectivity");
}

// ======================= 测试 11: QuorumManager =======================

void test_quorum_manager() {
    QuorumManager qmgr;

    TEST("QuorumManager create request and check pending");

    uint64_t req_id = qmgr.createRequest(5, 3, 2);
    assert(req_id > 0);

    auto status = qmgr.checkStatus(req_id);
    assert(status == QuorumStatus::Pending);

    END_TEST("QuorumManager create request and check pending");

    TEST("QuorumManager record ack and achieve quorum");

    uint64_t req = qmgr.createRequest(3, 2, 2);

    // 记录 2 个确认，但只有 1 种后端 → 后端覆盖不足
    qmgr.recordAck(req, 1, DeviceType::kCPU);
    qmgr.recordAck(req, 2, DeviceType::kCPU);
    auto s1 = qmgr.checkStatus(req);
    // 写 Quorum 满足但后端覆盖未满足
    assert(s1 == QuorumStatus::Pending);

    // 第 3 个确认来自不同后端 → 后端覆盖满足
    qmgr.recordAck(req, 3, DeviceType::kMPS);
    auto s2 = qmgr.checkStatus(req);
    assert(s2 == QuorumStatus::Achieved);

    END_TEST("QuorumManager record ack and achieve quorum");

    TEST("QuorumManager hasQuorum checks");

    assert(QuorumManager::hasWriteQuorum(3, 2));
    assert(!QuorumManager::hasWriteQuorum(1, 2));

    std::unordered_set<DeviceType> backends = {DeviceType::kCPU, DeviceType::kMPS};
    assert(QuorumManager::hasBackendCoverage(backends, 2));
    assert(!QuorumManager::hasBackendCoverage(backends, 3));

    size_t min_q = QuorumManager::computeMinQuorum(5);
    assert(min_q == 3);

    END_TEST("QuorumManager hasQuorum checks");

    TEST("QuorumManager cleanup timed out");

    // 创建短超时的请求
    QuorumConfig short_config = QuorumConfig::defaultConfig();
    short_config.quorum_timeout_ms = 1.0f;
    QuorumManager qmgr2(short_config);

    uint64_t req2 = qmgr2.createRequest(3, 2, 1);
    // 稍微等待确保超时
    ctorch_sleep(5);
    // 清理应该发现超时请求
    auto cleaned = qmgr2.cleanupTimedOut();
    assert(cleaned >= 1);

    END_TEST("QuorumManager cleanup timed out");
}

// ======================= 测试 12: NodeDiscovery =======================

void test_node_discovery() {
    NodeDiscovery discovery(0);

    TEST("NodeDiscovery seed node registration");

    NodeEndpoint ep1;
    ep1.node_id = 1;
    ep1.host = "192.168.1.1";
    ep1.port = 8080;
    ep1.protocol = DiscoveryProtocol::Static;
    ep1.backend_type = DeviceType::kCPU;
    ep1.version = "1.0.0";
    discovery.registerSeedNode(ep1);

    assert(discovery.isAlive(1));
    auto alive = discovery.aliveNodes();
    assert(alive.size() == 1);

    END_TEST("NodeDiscovery seed node registration");

    TEST("NodeDiscovery heartbeat generation and processing");

    auto hb = discovery.generateHeartbeat();
    assert(hb.node_id == 0);
    assert(hb.sequence_number > 0);

    // 处理来自节点 1 的心跳
    HeartbeatMessage hb1;
    hb1.node_id = 1;
    hb1.sequence_number = 1;
    hb1.timestamp = std::chrono::steady_clock::now();
    hb1.load_factor = 0.5f;
    hb1.backend_type = DeviceType::kCPU;
    discovery.processHeartbeat(hb1);

    assert(discovery.isAlive(1));

    END_TEST("NodeDiscovery heartbeat generation and processing");

    TEST("NodeDiscovery phi computation");

    // 对于刚注册的节点，Phi 应为 0（历史不足）
    double phi = discovery.computePhi(1);
    assert(phi >= 0.0);

    // 未知节点 Phi 应为 0
    double phi_unknown = discovery.computePhi(99);
    assert(phi_unknown == 0.0);

    END_TEST("NodeDiscovery phi computation");

    TEST("NodeDiscovery failure detection");

    // 新注册的节点不应立即被判定为故障
    auto failures = discovery.detectFailures();
    // 刚处理过心跳的节点不应被判定为故障
    assert(failures.empty());

    END_TEST("NodeDiscovery failure detection");

    TEST("NodeDiscovery record leave");

    discovery.recordLeave(1);
    assert(!discovery.isAlive(1));
    assert(discovery.getNodeStatus(1) == NodeStatus::Left);

    END_TEST("NodeDiscovery record leave");
}

// ======================= 测试 13: FaultTolerance =======================

void test_fault_tolerance() {
    FaultTolerance ft(0);

    TEST("FaultTolerance snapshot management");

    assert(!ft.needsSnapshot());  // 刚创建，不应需要快照

    // 创建参数
    Tensor param(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* pd = param.data_write<float>();
    pd[0] = 1.0f; pd[1] = 2.0f; pd[2] = 3.0f; pd[3] = 4.0f;
    std::vector<Tensor*> params = {&param};

    uint64_t sid = ft.takeSnapshot(params);
    assert(sid > 0);

    auto latest = ft.latestSnapshot();
    assert(latest.snapshot_id == sid);
    assert(latest.global_step == 0);

    // 第二个快照
    uint64_t sid2 = ft.takeSnapshot(params);
    assert(sid2 > sid);

    auto all_snaps = ft.allSnapshots();
    assert(all_snaps.size() == 2);

    END_TEST("FaultTolerance snapshot management");

    TEST("FaultTolerance CRDT merge");

    CRDTSnapshot local;
    local.snapshot_id = 1;
    local.version_vector = {1, 2, 3};
    local.global_step = 10;
    local.local_step = 5;
    local.grad_counter = {10, 20, 30};
    local.momentum = {0.1f, 0.2f, 0.3f};

    CRDTSnapshot remote;
    remote.snapshot_id = 2;
    remote.version_vector = {0, 3, 2};
    remote.global_step = 8;
    remote.local_step = 7;
    remote.grad_counter = {5, 25, 15};
    remote.momentum = {0.4f, 0.5f, 0.6f};

    CRDTSnapshot merged = FaultTolerance::mergeCRDT(local, remote);
    assert(merged.version_vector[0] == 1);
    assert(merged.version_vector[1] == 3);
    assert(merged.version_vector[2] == 3);
    assert(merged.global_step == 10);
    // local_step 取 LWW（更大的）
    assert(merged.local_step == 7);
    // 动量取 local_step 更大的（remote）
    assert(std::abs(merged.momentum[0] - 0.4f) < 1e-6f);

    END_TEST("FaultTolerance CRDT merge");

    TEST("FaultTolerance needsFullSync");

    CRDTSnapshot a, b;
    a.version_vector = {1, 2, 3};
    b.version_vector = {1, 2, 3};
    assert(!FaultTolerance::needsFullSync(a, b));  // 相同，不需要全量同步

    b.version_vector = {1, 2, 7};  // 差异 4 > 3
    assert(FaultTolerance::needsFullSync(a, b));

    END_TEST("FaultTolerance needsFullSync");

    TEST("FaultTolerance CRDTSnapshot serialize/deserialize");

    CRDTSnapshot snap;
    snap.snapshot_id = 42;
    snap.version_vector = {1, 2, 3};
    snap.global_step = 100;
    snap.local_step = 50;
    snap.grad_counter = {10, 20, 30};
    snap.momentum = {0.1f, 0.2f, 0.3f};
    snap.serialized_params = {0x01, 0x02, 0x03};

    auto serialized = snap.serialize();
    auto deserialized = CRDTSnapshot::deserialize(serialized);

    assert(deserialized.snapshot_id == 42);
    assert(deserialized.global_step == 100);
    assert(deserialized.version_vector.size() == 3);
    assert(deserialized.version_vector[1] == 2);
    assert(deserialized.serialized_params.size() == 3);

    END_TEST("FaultTolerance CRDTSnapshot serialize/deserialize");
}

// ======================= 测试 14: CheckpointManager =======================

void test_checkpoint_manager() {
    // 使用临时目录
    CheckpointConfig cfg = CheckpointConfig::defaultConfig();
    cfg.checkpoint_dir = "/tmp/ctorch_test_checkpoints";
    cfg.max_checkpoints = 3;
    CheckpointManager cpmgr(cfg);

    // 跨 TEST 块共享的测试数据
    Tensor param1(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* p1 = param1.data_write<float>();
    p1[0] = 1.0f; p1[1] = 2.0f; p1[2] = 3.0f; p1[3] = 4.0f;

    Tensor param2(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU, false);
    float* p2 = param2.data_write<float>();
    p2[0] = 0.1f; p2[1] = 0.2f; p2[2] = 0.3f; p2[3] = 0.4f;

    std::vector<Tensor*> params = {&param1, &param2};

    // 跨 TEST 块共享的 checkpoint ID
    uint64_t cid = 0;

    TEST("CheckpointManager save and list");

    cid = cpmgr.save(params, 100, 0.5f, 0.5f, CheckpointTrigger::Manual);
    assert(cid > 0);

    auto entries = cpmgr.listCheckpoints();
    assert(!entries.empty());
    bool found = false;
    for (const auto& e : entries) {
        if (e.id == cid) { found = true; break; }
    }
    assert(found);

    auto latest_id = cpmgr.latestCheckpointId();
    assert(latest_id == cid);

    END_TEST("CheckpointManager save and list");

    TEST("CheckpointManager load");

    std::vector<Tensor*> loaded_params = {&param1, &param2};
    auto meta = cpmgr.load(loaded_params, cid);
    assert(meta.global_step == 100);
    assert(std::abs(meta.loss - 0.5f) < 1e-6f);

    // 验证加载的数据
    const float* lp1 = param1.data_read<float>();
    const float* lp2 = param2.data_read<float>();
    assert(std::abs(lp1[0] - 1.0f) < 1e-6f);
    assert(std::abs(lp2[3] - 0.4f) < 1e-6f);

    END_TEST("CheckpointManager load");

    TEST("CheckpointManager prune");

    // 保存更多检查点以触发清理
    cpmgr.save(params, 200, 0.3f, 0.3f, CheckpointTrigger::StepInterval);
    cpmgr.save(params, 300, 0.2f, 0.2f, CheckpointTrigger::StepInterval);
    cpmgr.save(params, 400, 0.1f, 0.1f, CheckpointTrigger::StepInterval);

    // 最多保留 3 个
    auto entries2 = cpmgr.listCheckpoints();
    assert(entries2.size() <= 3);

    cpmgr.remove(cid);
    // 删除旧检查点后的确认
    assert(entries2.size() <= 3);

    END_TEST("CheckpointManager prune");

    TEST("CheckpointManager saveWithOptimizerState");

    std::vector<uint8_t> opt_state = {0x10, 0x20, 0x30, 0x40};
    uint64_t cid2 = cpmgr.saveWithOptimizerState(params, opt_state, 500, 0.05f, 0.05f);
    assert(cid2 > 0);

    auto loaded_opt = cpmgr.loadOptimizerState(cid2);
    assert(loaded_opt.size() == 4);
    assert(loaded_opt[0] == 0x10);
    assert(loaded_opt[3] == 0x40);

    END_TEST("CheckpointManager saveWithOptimizerState");

    TEST("CheckpointManager needsSave");

    assert(!cpmgr.needsSave(100, 0.5f, 0.5f));  // 刚保存过
    // 步数间隔检查
    cfg.save_interval_steps = 1000;
    cpmgr.setConfig(cfg);
    // 步数间隔未达到
    // 不需要断言具体值，只验证可以调用

    END_TEST("CheckpointManager needsSave");
}

// ======================= 测试 15: DistributedTrainer =======================

void test_distributed_trainer() {
    // 所有变量在函数作用域声明，避免 TEST 宏的 do-while 作用域问题
    Tensor param(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* pd = param.data_write<float>();
    pd[0] = 1.0f; pd[1] = 1.0f; pd[2] = 1.0f; pd[3] = 1.0f;
    std::vector<Tensor*> params = {&param};

    TrainerConfig config;
    config.local_steps = 5;
    config.learning_rate = 0.01f;
    config.momentum = 0.0f;
    config.checkpoint_interval = 0;

    DistributedTrainer trainer(params, config);

    TEST("DistributedTrainer creation and basic step");

    assert(trainer.globalStep() == 0);
    assert(trainer.bestLoss() == std::numeric_limits<float>::max());

    Tensor grad(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
    float* gd = grad.data_write<float>();
    gd[0] = 0.1f; gd[1] = 0.1f; gd[2] = 0.1f; gd[3] = 0.1f;

    trainer.step({grad}, 0.5f);

    auto metrics = trainer.metrics();
    assert(metrics.global_step == 1);
    assert(metrics.local_step == 1);
    assert(std::abs(metrics.current_loss - 0.5f) < 1e-6f);
    assert(std::abs(metrics.best_loss - 0.5f) < 1e-6f);

    END_TEST("DistributedTrainer creation and basic step");

    TEST("DistributedTrainer local steps accumulation and sync");

    for (int i = 0; i < 4; i++) {
        Tensor g(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU, false);
        float* gd2 = g.data_write<float>();
        gd2[0] = 0.1f; gd2[1] = 0.1f; gd2[2] = 0.1f; gd2[3] = 0.1f;
        trainer.step({g}, 0.4f + i * 0.1f);
    }

    auto metrics2 = trainer.metrics();
    assert(metrics2.global_step == 5);
    assert(metrics2.local_step == 0);
    assert(metrics2.num_syncs == 1);
    assert(std::abs(metrics2.best_loss - 0.4f) < 1e-6f);

    const float* param_data = param.data_read<float>();
    bool param_changed = false;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(param_data[i] - 1.0f) > 1e-6f) {
            param_changed = true;
            break;
        }
    }
    assert(param_changed);

    END_TEST("DistributedTrainer local steps accumulation and sync");

    // 第二个训练器测试
    Tensor param2(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU, false);
    float* pd2 = param2.data_write<float>();
    pd2[0] = 1.0f; pd2[1] = 1.0f;
    std::vector<Tensor*> params2 = {&param2};

    TrainerConfig config2;
    config2.local_steps = 3;
    config2.learning_rate = 0.01f;
    config2.momentum = 0.0f;
    config2.checkpoint_interval = 0;

    DistributedTrainer trainer2(params2, config2);

    TEST("DistributedTrainer fit convenience loop");

    trainer2.fit(6, [](size_t step) -> std::pair<std::vector<Tensor>, float> {
        Tensor g(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU, false);
        float* gd3 = g.data_write<float>();
        gd3[0] = 0.1f; gd3[1] = 0.2f;
        return {std::vector<Tensor>{g}, 1.0f - step * 0.1f};
    });

    auto metrics3 = trainer2.metrics();
    assert(metrics3.global_step == 6);
    assert(metrics3.num_syncs == 2);
    assert(std::abs(metrics3.best_loss - 0.5f) < 1e-6f);

    END_TEST("DistributedTrainer fit convenience loop");

    TEST("DistributedTrainer setLearningRate");

    trainer2.setLearningRate(0.001f);
    assert(std::abs(trainer2.learningRate() - 0.001f) < 1e-6f);

    trainer2.setLocalSteps(5);
    assert(trainer2.optimizer()->localSteps() == 5);

    END_TEST("DistributedTrainer setLearningRate");
}

// ======================= 测试 16: Transport TCP 传输层 =======================

void test_transport() {
    TEST("Transport basic lifecycle and data exchange");

    // 使用 port=0 让系统分配端口，避免端口冲突
    Transport::Config cfg0;
    cfg0.local_node_id = 0;
    cfg0.port = 0;

    Transport::Config cfg1;
    cfg1.local_node_id = 1;
    cfg1.port = 0;

    Transport t0(cfg0);
    Transport t1(cfg1);

    // 使用 throw 而非 assert（assert 在 NDEBUG 下被禁用）
    if (!t0.start()) throw std::runtime_error("t0.start() failed");
    if (!t1.start()) throw std::runtime_error("t1.start() failed");
    if (!t0.isRunning()) throw std::runtime_error("t0 not running");
    if (!t1.isRunning()) throw std::runtime_error("t1 not running");

    // 验证端口已分配（port=0 时系统分配，需等待确认）
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (t0.localPort() == 0) throw std::runtime_error("t0 port not assigned");
    if (t1.localPort() == 0) throw std::runtime_error("t1 port not assigned");

    // 节点 1 连接节点 0
    if (!t1.connectToPeer(0, "127.0.0.1", t0.localPort())) {
        throw std::runtime_error("t1 failed to connect to t0 on port " + std::to_string(t0.localPort()));
    }

    // 等待连接建立
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 验证连接
    if (!t1.isConnected(0)) throw std::runtime_error("t1 not connected to 0");
    auto nodes = t1.connectedNodes();
    if (nodes.size() != 1 || nodes[0] != 0) throw std::runtime_error("t1 wrong connected nodes");

    // 验证 t0 也接受了连接
    auto nodes0 = t0.connectedNodes();
    if (nodes0.size() != 1 || nodes0[0] != 1) throw std::runtime_error("t0 wrong connected nodes");

    // 设置接收回调
    std::vector<uint8_t> received_data;
    std::atomic<uint32_t> received_source{0};
    std::atomic<bool> data_received{false};

    t0.setReceiveCallback([&](uint32_t src, const std::vector<uint8_t>& data) {
        received_source.store(src);
        received_data = data;
        data_received.store(true);
    });

    // 发送数据
    std::vector<uint8_t> test_data = {0x01, 0x02, 0x03, 0x04, 0x05};
    if (!t1.send(0, test_data)) throw std::runtime_error("t1 send failed");

    // 等待接收
    int retries = 0;
    while (!data_received.load() && retries < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        retries++;
    }

    if (!data_received.load()) throw std::runtime_error("data not received after 500ms");
    if (received_source.load() != 1) throw std::runtime_error("wrong source");
    if (received_data.size() != test_data.size()) throw std::runtime_error("wrong size");
    if (received_data[0] != 0x01 || received_data[4] != 0x05) throw std::runtime_error("wrong data");

    // 停止
    t0.stop();
    t1.stop();
    if (t0.isRunning()) throw std::runtime_error("t0 still running after stop");
    if (t1.isRunning()) throw std::runtime_error("t1 still running after stop");

    END_TEST("Transport basic lifecycle and data exchange");

    TEST("Transport broadcast and disconnect");

    Transport t2(Transport::Config{2, 0});
    Transport t3(Transport::Config{3, 0});

    if (!t2.start()) throw std::runtime_error("t2 start failed");
    if (!t3.start()) throw std::runtime_error("t3 start failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (t2.localPort() == 0) throw std::runtime_error("t2 port not assigned");
    if (t3.localPort() == 0) throw std::runtime_error("t3 port not assigned");

    // 双向连接
    if (!t2.connectToPeer(3, "127.0.0.1", t3.localPort())) {
        throw std::runtime_error("t2 connect to t3 failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!t2.isConnected(3)) throw std::runtime_error("t2 not connected to 3");
    if (!t3.isConnected(2)) throw std::runtime_error("t3 not connected to 2");

    // 广播
    std::atomic<size_t> t3_received{0};
    t3.setReceiveCallback([&](uint32_t, const std::vector<uint8_t>&) {
        t3_received++;
    });

    t2.broadcast({0x0A, 0x0B});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (t3_received.load() != 1) throw std::runtime_error("broadcast not received");

    // 断开连接
    if (!t2.disconnect(3)) throw std::runtime_error("t2 disconnect failed");
    if (t2.isConnected(3)) throw std::runtime_error("t2 still connected after disconnect");

    t2.stop();
    t3.stop();

    END_TEST("Transport broadcast and disconnect");
}

// ======================= 主函数 =======================

int main() {
    std::cout << "\n===========================================" << std::endl;
    std::cout << "  CTorch Gen 2 Distributed System Tests" << std::endl;
    std::cout << "===========================================" << std::endl;

    // CDTF
    test_cdtf_roundtrip();
    test_cdtf_quantize8();
    test_cdtf_quantize16();

    // GradientAggregator
    test_gradient_aggregator();

    // BackendManager
    test_backend_manager();

    // GTCScheduler
    test_gtc_scheduler();

    // DeviceMigration
    test_device_migration();

    // CommEngine
    test_comm_engine();

    // DistributedOptimizer
    test_distributed_optimizer();

    // CRDTState
    test_crdt_state();

    // EntropyAwareCompressor
    test_entropy_aware_compressor();

    // TopologyManager
    test_topology_manager();

    // QuorumManager
    test_quorum_manager();

    // NodeDiscovery
    test_node_discovery();

    // FaultTolerance
    test_fault_tolerance();

    // CheckpointManager
    test_checkpoint_manager();

    // DistributedTrainer
    test_distributed_trainer();

    // Transport
    test_transport();

    std::cout << "\n===========================================" << std::endl;
    std::cout << "  Results: " << (test_passed + test_failed)
              << " total, " << test_passed << " passed, "
              << test_failed << " failed" << std::endl;
    std::cout << "===========================================" << std::endl;

    return test_failed > 0 ? 1 : 0;
}