// RUN: %clang_cc1 -verify -fompss-2 -x c++ -fexceptions -fcxx-exceptions %s -Wuninitialized
int main() {
  auto l = []() -> int * { return nullptr; };
  #pragma oss task in(l()[30]) // expected-error {{call expressions are not supported}}
  {}
}
