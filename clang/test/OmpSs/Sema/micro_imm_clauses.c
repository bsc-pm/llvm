// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

#pragma oss task immediate(1) // expected-error {{unexpected OmpSs-2 clause 'immediate' in directive '#pragma oss task'}}
void foo();
#pragma oss task microtask(1) // expected-error {{unexpected OmpSs-2 clause 'microtask' in directive '#pragma oss task'}}
void bar();

int main() {
  #pragma oss task immediate(1) // expected-error {{unexpected OmpSs-2 clause 'immediate' in directive '#pragma oss task'}}
  {}
  #pragma oss task microtask(1) // expected-error {{unexpected OmpSs-2 clause 'microtask' in directive '#pragma oss task'}}
  {}
}
