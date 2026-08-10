#include "test_mps_flush.h"
#include "src/kernels/kernels.h"

void test_mps_flush_wait() {
    MPS_flush(true);
}
