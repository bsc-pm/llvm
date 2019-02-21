// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

int main() {
  int x;
  #pragma oss task default(asdf) // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  #pragma oss task default(shared)
  #pragma oss task default(none)
  #pragma oss task default(none) default(shared) // expected-error {{directive '#pragma oss task' cannot contain more than one 'default' clause}}
  {}
  #pragma oss task default(none)
  #pragma oss task default(none) depend(in: x) // expected-error {{expected explicit data-sharing for 'x'}}
  { ++x; ++x; } // expected-error {{expected explicit data-sharing for 'x'}}
}
