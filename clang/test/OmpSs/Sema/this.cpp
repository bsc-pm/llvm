// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o -

struct S {
  int a;
  S() {}
  void foo();
};

void S::foo() {
    auto l = []() { // expected-note {{explicitly capture 'this'}}
        #pragma oss task
        { a++; } // expected-error {{'this' cannot be implicitly captured in this context}}
    };
    auto p = [this]() {
        #pragma oss task
        { a++; }
    };
}
