// RUN: %clang_cc1 -verify -fopenmp %s
// expected-no-diagnostics
int main() {
  #pragma omp task label("1")
  {}
  #pragma omp parallel for label("2")
  for (int i = 0; i < 10; ++i)
  {}
  #pragma omp parallel
  #pragma omp for label("3")
  for (int i = 0; i < 10; ++i)
  {}
  #pragma omp task default(none) label("1")
  {}
  #pragma omp parallel for default(none) label("2")
  for (int i = 0; i < 10; ++i)
  {}
  #pragma omp parallel default(none)
  #pragma omp for label("3")
  for (int i = 0; i < 10; ++i)
  {}
}
