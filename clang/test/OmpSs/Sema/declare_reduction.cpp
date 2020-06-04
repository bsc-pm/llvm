// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
// expected-no-diagnostics

// We want to check copy constructor.
#pragma oss declare reduction(direct: int: omp_out) initializer(omp_priv(0))

