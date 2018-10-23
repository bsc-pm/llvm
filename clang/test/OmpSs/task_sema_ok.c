// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -fsyntax-only
// expected-no-diagnostics

int main(void) {
  #pragma oss task
  {}
}

