// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -std=c++11 %s

// This test has been ported from clang/test/OpenMP/openmp_check.cpp

// Ensure SkipUntil behaviour is not modified when using it in a non-OmpSs-2
// place. That is, OmpSs-2 tokens are skipped when Parser is not in OmpSs-2
// parsing mode.

#define p _Pragma("oss task")

int nested(int a) {
#pragma oss task p // expected-error {{unexpected OmpSs-2 directive}}
  ++a;
#pragma oss task
  ++a;

  auto F = [&]() {
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-3 {{expected expression}}
#endif

#pragma oss task
    {
      ++a;
    }
  };
  F();
  return a;
}
