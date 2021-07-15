// RUN: %clang_cc1 -verify -x c++ -std=c++11 -fompss-2 -ferror-limit 100 -o - %s
// expected-no-diagnostics

// The constexpr template function instantiates inside the task context.
// This test does not expect a diagnostic about foo's return.

template<typename T> constexpr T foo(T t) { return t; }

int main() {
    #pragma oss task
    { int a = foo(3); }
}
