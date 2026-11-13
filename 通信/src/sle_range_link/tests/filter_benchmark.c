#include "range_link_filter.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

/* Batches reduce timer overhead; identical deterministic inputs in both builds. */
int main(void)
{
    uint32_t seed = 20261113U, count = 0, batch, i;
    uint64_t checksum = 0;
    rl_filter_init();
    for (batch = 0; batch < 55; ++batch) {
        clock_t start = clock();
        for (i = 0; i < 100000; ++i) {
            seed = seed * 1664525U + 1013904223U;
            checksum += (uint32_t)rl_filter_push(10000 + (int32_t)(seed % 401U) - 200, &count);
        }
        if (batch >= 5) printf("%.9f\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    }
    fprintf(stderr, "checksum=%" PRIu64 " count=%" PRIu32 "\n", checksum, count);
    return 0;
}
