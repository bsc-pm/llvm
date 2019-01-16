// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int main(void) {
  #pragma oss taskwait asdf // expected-warning {{extra tokens at the end of '#pragma oss taskwait' are ignored}}
}

