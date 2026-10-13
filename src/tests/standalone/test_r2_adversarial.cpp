/**
 * @file test_r2_adversarial.cpp
 * @brief Empirical Adversarial Challenge Suite for Requirement R2:
 *        C3 L1 Lockless Dispatch Table Lifecycle & ABA Elimination
 * @details Adversarial stress vectors:
 *   1. High thread count reader/writer races (32 readers, 16 writers, 8 clearers).
 *   2. Slot index hash collision integrity (many distinct hashes mapping to same FastSlot).
 *   3. ABA immunity verification (slot overwritten and restored during reader flight).
 *   4. Zero-UAF lifecycle tracking during asynchronous uninstallation.
 *   5. Concurrent CAS write coordination under heavy slot contention.
 *   6. Boundary conditions (zero hash, null kernel, exception safety).
 *   7. Deterministic probe for stale L2 backfill demoting a promoted Tier 2 kernel.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <cassert>
#include <string>
#include <condition_variable>
#include <mutex>
#include <random>

#include "c3/include/C3/C3Engine.h"
#include "c3/include/C3/C3KernelRegistry.h"
#include "Tensor.h"

using namespace ct;
using namespace c3;

static std::atomic<int> g_adversarial_fails{0};
static std::atomic<int> g_adversarial_passes{0};

#define ADV_CHECK(cond, msg)                                                    \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "FAIL [Adversarial R2]: " << msg << "\n";              \
            g_adversarial_fails.fetch_add(1, std::memory_order_relaxed);        \
        } else {                                                                \
            std::cout << "PASS [Adversarial R2]: " << msg << "\n";              \
            g_adversarial_passes.fetch_add(1, std::memory_order_relaxed);       \
        }                                                                       \
    } while (0)

// ============================================================================
// Mock CompiledKernel implementations for adversarial testing
// ============================================================================

class AdvMockKernel : public CompiledKernel {
public:
    AdvMockKernel(int opt_lvl, std::string name, uint64_t magic_id = 0)
        : name_(std::move(name)), magic_id_(magic_id) {
        opt_level_ = opt_lvl;
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        exec_count_.fetch_add(1, std::memory_order_relaxed);
        if (inputs.empty()) return {};
        // Output tensor matching first input
        Tensor out = inputs[0].clone();
        if (inputs.size() >= 2 && inputs[0].numel() == inputs[1].numel()) {
            const float* pa = inputs[0].data_read<float>();
            const float* pb = inputs[1].data_read<float>();
            float* pout = out.data_write<float>();
            for (size_t i = 0; i < out.numel(); ++i) {
                pout[i] = pa[i] + pb[i];
            }
        }
        return {out};
    }

    const std::string& cacheKey() const override { return name_; }
    DeviceType targetDevice() const override { return DeviceType::kCPU; }
    size_t workspaceBytes() const override { return 0; }

    std::string name() const { return name_; }
    uint64_t magic_id() const { return magic_id_; }
    size_t execCount() const { return exec_count_.load(std::memory_order_relaxed); }

protected:
    std::string name_;
    uint64_t magic_id_{0};
    std::atomic<size_t> exec_count_{0};
};

class LifecycleTrackingKernel : public AdvMockKernel {
public:
    LifecycleTrackingKernel(int opt_lvl, std::string name, std::shared_ptr<std::atomic<bool>> destroyed_flag)
        : AdvMockKernel(opt_lvl, std::move(name)), destroyed_flag_(std::move(destroyed_flag)) {}

    ~LifecycleTrackingKernel() override {
        if (destroyed_flag_) {
            destroyed_flag_->store(true, std::memory_order_release);
        }
    }

private:
    std::shared_ptr<std::atomic<bool>> destroyed_flag_;
};

// ============================================================================
// Test 1: Zero Hash & Null Kernel Boundary Invariants
// ============================================================================
void test_boundary_invariants() {
    std::cout << "\n--- [Test 1] Zero Hash & Null Kernel Boundary Invariants ---\n";
    auto& registry = C3KernelRegistry::getInstance();
    registry.uninstallAll();

    // 1. Zero hash lookup
    auto res_zero = registry.lookupFast(0);
    ADV_CHECK(res_zero == nullptr, "lookupFast(0) returns nullptr");

    // 2. Install zero hash (should be rejected/ignored)
    auto k = std::make_shared<AdvMockKernel>(2, "zero_test");
    registry.installFast(0, k);
    ADV_CHECK(registry.lookupFast(0) == nullptr, "installFast with hash=0 is safely ignored");

    // 3. Install null kernel (should be rejected/ignored)
    uint64_t hash = 0xdeadbeefULL;
    registry.installFast(hash, nullptr);
    ADV_CHECK(registry.lookupFast(hash) == nullptr, "installFast with null kernel is safely ignored");

    // 4. Install valid kernel and verify
    registry.installFast(hash, k);
    auto retrieved = registry.lookupFast(hash);
    ADV_CHECK(retrieved != nullptr && retrieved.get() == k.get(), "Valid installFast retrieved successfully");

    // 5. Clear fast slots
    registry.clearFastSlots();
    ADV_CHECK(registry.lookupFast(hash) == nullptr, "clearFastSlots safely invalidated entry");
}

// ============================================================================
// Test 2: FastSlot Hash Collision & Cross-Slot Isolation
// 64 distinct hashes all mapping to the exact same slot index (hash & 2047)
// ============================================================================
void test_hash_collision_isolation() {
    std::cout << "\n--- [Test 2] FastSlot Hash Collision Isolation & Integrity ---\n";
    auto& registry = C3KernelRegistry::getInstance();
    registry.uninstallAll();

    constexpr size_t kCollisions = 64;
    constexpr size_t kSlotIndex = 777; // target slot
    std::vector<uint64_t> hashes;
    std::vector<std::shared_ptr<AdvMockKernel>> kernels;

    for (size_t i = 0; i < kCollisions; ++i) {
        uint64_t h = kSlotIndex + (i + 1) * C3KernelRegistry::kFastTableSize;
        hashes.push_back(h);
        kernels.push_back(std::make_shared<AdvMockKernel>(static_cast<int>(i + 1), "coll_" + std::to_string(i), h));
    }

    // Concurrent writers and readers targeting the exact same slot with different hashes
    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> successful_lookups{0};
    std::atomic<size_t> wrong_kernel_detected{0};

    // 8 reader threads
    std::vector<std::thread> readers;
    for (int r = 0; r < 8; ++r) {
        readers.emplace_back([&, r]() {
            std::mt19937_64 rng(1337 + r);
            while (!stop_flag.load(std::memory_order_relaxed)) {
                size_t idx = rng() % kCollisions;
                uint64_t target_hash = hashes[idx];
                auto k = registry.lookupFast(target_hash);
                if (k) {
                    auto* adv_k = dynamic_cast<AdvMockKernel*>(k.get());
                    if (!adv_k || adv_k->magic_id() != target_hash) {
                        wrong_kernel_detected.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        successful_lookups.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // 4 writer threads continuously overwriting the colliding slot
    std::vector<std::thread> writers;
    for (int w = 0; w < 4; ++w) {
        writers.emplace_back([&, w]() {
            std::mt19937_64 rng(7331 + w);
            for (int iter = 0; iter < 10000; ++iter) {
                size_t idx = rng() % kCollisions;
                registry.installFast(hashes[idx], kernels[idx]);
            }
        });
    }

    for (auto& t : writers) t.join();
    stop_flag.store(true, std::memory_order_release);
    for (auto& t : readers) t.join();

    ADV_CHECK(wrong_kernel_detected.load() == 0,
              "Zero cross-slot hash collision corruptions detected across colliding hashes");
    ADV_CHECK(successful_lookups.load() > 0,
              "At least one valid concurrent lookup succeeded during collision stress");
    registry.uninstallAll();
}

// ============================================================================
// Test 3: Zero-UAF Lifecycle Verification (Thread Safety Across Uninstall)
// ============================================================================
void test_zero_uaf_destruction_tracking() {
    std::cout << "\n--- [Test 3] Zero-UAF Lifecycle Tracking Across Asynchronous Clear ---\n";
    auto& registry = C3KernelRegistry::getInstance();
    registry.uninstallAll();

    auto destroyed = std::make_shared<std::atomic<bool>>(false);
    uint64_t test_hash = 0xabcdef01ULL;

    {
        auto k = std::make_shared<LifecycleTrackingKernel>(2, "tracked_kernel", destroyed);
        registry.installFast(test_hash, k);
    } // k local variable drops out of scope; only registry fast slot holds it

    ADV_CHECK(!destroyed->load(), "Kernel remains alive while held in FastSlot");

    // Reader thread acquires strong shared_ptr reference
    std::shared_ptr<CompiledKernel> acquired = registry.lookupFast(test_hash);
    ADV_CHECK(acquired != nullptr, "Reader successfully acquired shared_ptr from FastSlot");

    // Asynchronously clear fast slots and uninstall
    registry.clearFastSlots();
    registry.uninstallAll();

    // Verify FastSlot has dropped its reference, but acquired is STILL alive
    ADV_CHECK(!destroyed->load(), "Kernel remains alive while reader holds acquired shared_ptr");

    // Reader executes safely
    Tensor a(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    auto outs = acquired->execute({a, b});
    ADV_CHECK(!outs.empty(), "Reader safely executed kernel post-uninstallation without crash");

    // Reader releases acquired shared_ptr
    acquired.reset();
    ADV_CHECK(destroyed->load(), "Kernel safely destructed after all references released (no leak / no UAF)");
}

// ============================================================================
// Test 4: Extreme Reader / Writer / Clearer Concurrency Stress
// 32 Readers, 16 Writers, 8 Clearers hammers table simultaneously
// ============================================================================
void test_extreme_multithreaded_concurrency() {
    std::cout << "\n--- [Test 4] Extreme Reader/Writer/Clearer Concurrency Stress (56 Threads) ---\n";
    auto& registry = C3KernelRegistry::getInstance();
    registry.uninstallAll();

    constexpr int kNumReaders = 32;
    constexpr int kNumWriters = 16;
    constexpr int kNumClearers = 8;
    constexpr size_t kNumDistinctHashes = 256;

    std::vector<uint64_t> hashes;
    std::vector<std::shared_ptr<AdvMockKernel>> kernels;
    for (size_t i = 0; i < kNumDistinctHashes; ++i) {
        uint64_t h = 0x10000000ULL + i * 17;
        hashes.push_back(h);
        kernels.push_back(std::make_shared<AdvMockKernel>(2, "kernel_" + std::to_string(i), h));
    }

    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> total_reads{0};
    std::atomic<size_t> valid_reads{0};
    std::atomic<size_t> total_writes{0};
    std::atomic<size_t> total_clears{0};
    std::atomic<size_t> corruption_errors{0};

    Tensor dummy(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);

    std::vector<std::thread> threads;

    // Launch readers
    for (int r = 0; r < kNumReaders; ++r) {
        threads.emplace_back([&, r]() {
            std::mt19937_64 rng(2000 + r);
            while (!stop_flag.load(std::memory_order_relaxed)) {
                size_t idx = rng() % kNumDistinctHashes;
                uint64_t target_hash = hashes[idx];
                auto k = registry.lookupFast(target_hash);
                total_reads.fetch_add(1, std::memory_order_relaxed);
                if (k) {
                    auto* adv_k = dynamic_cast<AdvMockKernel*>(k.get());
                    if (!adv_k || adv_k->magic_id() != target_hash) {
                        corruption_errors.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        valid_reads.fetch_add(1, std::memory_order_relaxed);
                        // Invoke execute safely
                        auto out = adv_k->execute({dummy});
                        if (out.empty()) {
                            corruption_errors.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
        });
    }

    // Launch writers
    for (int w = 0; w < kNumWriters; ++w) {
        threads.emplace_back([&, w]() {
            std::mt19937_64 rng(3000 + w);
            for (int iter = 0; iter < 5000; ++iter) {
                size_t idx = rng() % kNumDistinctHashes;
                registry.installFast(hashes[idx], kernels[idx]);
                total_writes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Launch clearers
    for (int c = 0; c < kNumClearers; ++c) {
        threads.emplace_back([&, c]() {
            for (int iter = 0; iter < 100; ++iter) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                registry.clearFastSlots();
                total_clears.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Join writers and clearers first
    for (size_t i = kNumReaders; i < threads.size(); ++i) {
        threads[i].join();
    }

    stop_flag.store(true, std::memory_order_release);
    for (int i = 0; i < kNumReaders; ++i) {
        threads[i].join();
    }

    std::cout << " - Total Reads   : " << total_reads.load() << " (Valid: " << valid_reads.load() << ")\n";
    std::cout << " - Total Writes  : " << total_writes.load() << "\n";
    std::cout << " - Total Clears  : " << total_clears.load() << "\n";

    ADV_CHECK(corruption_errors.load() == 0,
              "Zero corruption errors under 56-thread extreme reader/writer/clearer race");
    ADV_CHECK(valid_reads.load() > 0, "At least some reads successfully executed kernels");
    registry.uninstallAll();
}

// ============================================================================
// Test 5: ABA Challenge with Microsecond Interleaved Writes
// ============================================================================
void test_aba_elimination() {
    std::cout << "\n--- [Test 5] ABA Sequence Verification Under Alternating Slot Installs ---\n";
    auto& registry = C3KernelRegistry::getInstance();
    registry.uninstallAll();

    uint64_t sig_hash = 0x55555555ULL;
    auto kA = std::make_shared<AdvMockKernel>(1, "kernelA", sig_hash);
    auto kB = std::make_shared<AdvMockKernel>(2, "kernelB", sig_hash + 1);
    auto kA2 = std::make_shared<AdvMockKernel>(3, "kernelA_v2", sig_hash);

    // Initial state: install kA
    registry.installFast(sig_hash, kA);
    auto r1 = registry.lookupFast(sig_hash);
    ADV_CHECK(r1 != nullptr && r1->optLevel() == 1, "Initial slot installed with optLevel 1");

    // Overwrite with kB (different hash, colliding slot or same slot)
    registry.installFast(sig_hash, kB);
    // Overwrite back with kA2 (same hash as kA, but new kernel instance and new seq)
    registry.installFast(sig_hash, kA2);

    auto r2 = registry.lookupFast(sig_hash);
    ADV_CHECK(r2 != nullptr && r2->optLevel() == 3, "Slot holds newly installed kA2 with optLevel 3");
    ADV_CHECK(r2.get() != kA.get(), "ABA avoided: slot does not return old kA instance");
}

// ============================================================================
// Test 6: Deterministic Concurrency Probe:
// Does Stale L2 Execution Backfill Demote a Promoted Tier 2 Kernel in L1?
// ============================================================================
class LatchBlockingKernel : public CompiledKernel {
public:
    LatchBlockingKernel(int opt_lvl, std::string name,
                        std::shared_ptr<std::atomic<bool>> entered_latch,
                        std::shared_ptr<std::atomic<bool>> release_latch)
        : name_(std::move(name)),
          entered_latch_(std::move(entered_latch)), release_latch_(std::move(release_latch)) {
        opt_level_ = opt_lvl;
    }

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        if (entered_latch_) {
            entered_latch_->store(true, std::memory_order_release);
        }
        // Block until release_latch is signaled
        if (release_latch_) {
            while (!release_latch_->load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
        if (inputs.empty()) return {};
        return {inputs[0].clone()};
    }

    const std::string& cacheKey() const override { return name_; }
    DeviceType targetDevice() const override { return DeviceType::kCPU; }
    size_t workspaceBytes() const override { return 0; }

private:
    std::string name_;
    std::shared_ptr<std::atomic<bool>> entered_latch_;
    std::shared_ptr<std::atomic<bool>> release_latch_;
};

void test_tier_promotion_downgrade_probe() {
    std::cout << "\n--- [Test 6] Adversarial Probe: Concurrent Tier 1->2 Promotion vs L2 Backfill ---\n";
    auto& registry = C3KernelRegistry::getInstance();
    registry.uninstallAll();

    op target_op = op::Add;
    DeviceType dev = DeviceType::kCPU;
    KernelShapeInfo shapes;
    shapes.lhs_shape = {4, 4};
    shapes.rhs_shape = {4, 4};
    shapes.out_shape = {4, 4};

    uint64_t fast_hash = C3KernelRegistry::computeFastHash(target_op, dev, shapes.lhs_shape, shapes.rhs_shape);

    auto entered_latch = std::make_shared<std::atomic<bool>>(false);
    auto release_latch = std::make_shared<std::atomic<bool>>(false);

    // 1. Install Tier 1 kernel (optLevel = 2)
    auto k_tier1 = std::make_shared<LatchBlockingKernel>(2, "tier1_blocking", entered_latch, release_latch);
    registry.install(target_op, dev, k_tier1, shapes);

    // 2. Clear L1 fast slots to force the next tryExecute into L2 fallback
    registry.clearFastSlots();
    ADV_CHECK(registry.lookupFast(fast_hash) == nullptr, "L1 is cleared; lookupFast returns nullptr");

    Tensor a(ShapeTag{}, {4, 4}, DType::kFloat, dev);
    Tensor b(ShapeTag{}, {4, 4}, DType::kFloat, dev);

    // 3. Launch Thread T1 calling tryExecute. It will hit L2, load Tier 1, and block in execute()
    std::thread reader([&]() {
        auto res = registry.tryExecute(target_op, a, b);
        ADV_CHECK(res.has_value(), "tryExecute completed successfully in reader");
    });

    // Wait until T1 has entered execute() with Tier 1
    while (!entered_latch->load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::cout << " [Adversarial] Thread T1 is currently in-flight executing Tier 1 in L2.\n";

    // 4. Concurrently, compiler background thread installs Tier 2 kernel (optLevel = 4)
    auto k_tier2 = std::make_shared<AdvMockKernel>(4, "tier2_promoted");
    registry.install(target_op, dev, k_tier2, shapes);
    std::cout << " [Adversarial] Compiler thread installed Tier 2 kernel into registry.\n";

    // Check L1 right now: it should have Tier 2!
    auto mid_fast = registry.lookupFast(fast_hash);
    ADV_CHECK(mid_fast != nullptr && mid_fast->optLevel() == 4,
              "Right after promotion, L1 correctly contains Tier 2 kernel (optLevel 4)");

    // 5. Release Thread T1 so it completes its Tier 1 execution and performs L1 backfill
    std::cout << " [Adversarial] Releasing Thread T1 to finish and attempt L1 backfill...\n";
    release_latch->store(true, std::memory_order_release);
    reader.join();

    // 6. Inspect L1 post-execution!
    auto post_fast = registry.lookupFast(fast_hash);
    ADV_CHECK(post_fast != nullptr, "Post-execution L1 slot is non-null");

    if (post_fast) {
        int final_opt = post_fast->optLevel();
        std::cout << " [Adversarial] Post-execution L1 kernel optLevel is: " << final_opt << "\n";
        if (final_opt == 2) {
            std::cerr << " >>> CRITICAL DEFECT CONFIRMED: Stale L2 backfill demoted promoted Tier 2 (optLevel 4) to Tier 1 (optLevel 2)! <<<\n";
            g_adversarial_fails.fetch_add(1, std::memory_order_relaxed);
        } else if (final_opt == 4) {
            std::cout << " PASS: L1 retained Tier 2 kernel (optLevel 4) without stale backfill demotion!\n";
            g_adversarial_passes.fetch_add(1, std::memory_order_relaxed);
        } else {
            ADV_CHECK(false, "Unexpected optLevel in L1");
        }
    }

    registry.uninstallAll();
}

// ============================================================================
// Main Execution
// ============================================================================
int main() {
    std::cout << "================================================================================\n";
    std::cout << ">>> CTorch M2 (R2): Empirical Adversarial Challenge Suite <<<\n";
    std::cout << "================================================================================\n";

    test_boundary_invariants();
    test_hash_collision_isolation();
    test_zero_uaf_destruction_tracking();
    test_extreme_multithreaded_concurrency();
    test_aba_elimination();
    test_tier_promotion_downgrade_probe();

    std::cout << "\n================================================================================\n";
    std::cout << "Adversarial Challenge Results: "
              << g_adversarial_passes.load() << " PASSED, "
              << g_adversarial_fails.load() << " FAILED.\n";
    std::cout << "================================================================================\n";

    return (g_adversarial_fails.load() == 0) ? 0 : 1;
}
