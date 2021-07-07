// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

// cost/priority/onready clauses are not allowed to use
// loop iterators

int main() {
  #pragma oss taskloop collapse(2) cost(i + j) // expected-error 2 {{iterator usage in this clause is not allowed}}
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) {
      }
    }
  }
}
