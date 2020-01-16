// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

struct S {
  int x;
} s;

#pragma oss task if(s1) // expected-error {{statement requires expression of scalar type}}
void foo(struct S s1) {}

int main(void) {
  #pragma oss task if(s) // expected-error {{statement requires expression of scalar type}}
  {}
}
