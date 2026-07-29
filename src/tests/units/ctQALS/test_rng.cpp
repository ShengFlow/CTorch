
#include "../include/ctQALS/Random.h"
using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace ctQALS::rng;

template<typename Func>
double measure_time_us(Func&& func, int iterations = 10) {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, micro> diff = end - start;
    return diff.count() / iterations;
}

int main() {
    const size_t N = 10000000;
    const int ITERATIONS = 5;

    cout << "=== Random Number Generator Performance Test ===" << endl;
    cout << "Samples: " << N << ", Iterations: " << ITERATIONS << endl << endl;

    // ========== Xoshiro256++ + Ziggurat ==========
    {
        Xoshiro256PlusPlus engine(12345);
        ZigguratNormal normal(engine);
        vector<float> samples(N);

        double avg_time = measure_time_us([&]() {
            normal.fill(samples.data(), samples.size());
        }, ITERATIONS);

        double mean = 0.0, m2 = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) {
            double delta = samples[i] - mean;
            mean += delta / (i + 1);
            m2 += delta * (samples[i] - mean);
        }
        double stddev = sqrt(m2 / (samples.size() - 1));

        cout << "[Xoshiro256++ + Ziggurat]" << endl;
        cout << "  Avg Time: " << avg_time << " us" << endl;
        cout << "  Throughput: " << (N / avg_time * 1e6) << " samples/sec" << endl;
        cout << "  Mean: " << mean << " (expect ~0)" << endl;
        cout << "  Stddev: " << stddev << " (expect ~1)" << endl << endl;
    }

    // ========== mt19937_64 + normal_distribution ==========
    {
        mt19937_64 engine(12345);
        normal_distribution<double> dist(0.0, 1.0);
        vector<float> samples(N);

        double avg_time = measure_time_us([&]() {
            for (size_t i = 0; i < N; ++i) {
                samples[i] = static_cast<float>(dist(engine));
            }
        }, ITERATIONS);

        double mean = 0.0, m2 = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) {
            double delta = samples[i] - mean;
            mean += delta / (i + 1);
            m2 += delta * (samples[i] - mean);
        }
        double stddev = sqrt(m2 / (samples.size() - 1));

        cout << "[mt19937_64 + normal_distribution]" << endl;
        cout << "  Avg Time: " << avg_time << " us" << endl;
        cout << "  Throughput: " << (N / avg_time * 1e6) << " samples/sec" << endl;
        cout << "  Mean: " << mean << " (expect ~0)" << endl;
        cout << "  Stddev: " << stddev << " (expect ~1)" << endl << endl;
    }

    // ========== Xoshiro256++ uniform_f32 only ==========
    {
        Xoshiro256PlusPlus engine(12345);
        vector<float> samples(N);

        double avg_time = measure_time_us([&]() {
            for (size_t i = 0; i < N; ++i) {
                samples[i] = engine.uniform_f32();
            }
        }, ITERATIONS);

        cout << "[Xoshiro256++ uniform_f32 only]" << endl;
        cout << "  Avg Time: " << avg_time << " us" << endl;
        cout << "  Throughput: " << (N / avg_time * 1e6) << " samples/sec" << endl << endl;
    }

    // ========== mt19937_64 uniform only ==========
    {
        mt19937_64 engine(12345);
        vector<float> samples(N);

        double avg_time = measure_time_us([&]() {
            for (size_t i = 0; i < N; ++i) {
                samples[i] = static_cast<float>(engine() >> 11) * 0x1.0p-53f;
            }
        }, ITERATIONS);

        cout << "[mt19937_64 uniform only]" << endl;
        cout << "  Avg Time: " << avg_time << " us" << endl;
        cout << "  Throughput: " << (N / avg_time * 1e6) << " samples/sec" << endl << endl;
    }

    cout << "=== Test Complete ===" << endl;

    return 0;
}