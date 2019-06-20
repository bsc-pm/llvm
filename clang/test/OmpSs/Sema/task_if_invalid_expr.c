// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int main(void) {
  struct S {
      int x;
  } s;
  #pragma oss task if(s) // expected-error {{statement requires expression of scalar type}}
  {}
}
