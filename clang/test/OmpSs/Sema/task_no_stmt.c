// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int main(void) {
  #pragma oss task
  int x = 3; // expected-error {{expected expression}}
  #pragma oss task
} // expected-error {{expected statement}}
