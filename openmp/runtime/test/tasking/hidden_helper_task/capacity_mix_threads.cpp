// RUN: %libomp-cxx-compile-and-run
// UNSUPPORTED: ompv, ompv-free-agents, ompv-passive

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>

// AIX runs out of resource in 32-bit if 4*omp_get_max_threads() is more
// than 64 threads with the default stack size.
#if defined(_AIX) && !__LP64__
#define MAX_THREADS 64
#endif

void dummy_root() {
  // omp_get_max_threads() will do middle initialization
  int nthreads = omp_get_max_threads();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

int main(int argc, char *argv[]) {
  int N = std::min(std::max(std::max(32, 4 * omp_get_max_threads()),
                            4 * omp_get_num_procs()),
                   std::numeric_limits<int>::max());

#if defined(_AIX) && !__LP64__
  if (N > MAX_THREADS)
    N = MAX_THREADS;
#endif

  std::vector<int> data(N);

  // Create a new thread to initialize the OpenMP RTL. The new thread will not
  // be taken as the "initial thread".
  std::thread root(dummy_root);

#pragma omp parallel for num_threads(N)
  for (unsigned i = 0; i < N; ++i) {
    data[i] = i;
  }

#pragma omp parallel for num_threads(N + 1)
  for (unsigned i = 0; i < N; ++i) {
    data[i] += i;
  }

  for (unsigned i = 0; i < N; ++i) {
    assert(data[i] == 2 * i);
  }

  root.join();

  return 0;
}
