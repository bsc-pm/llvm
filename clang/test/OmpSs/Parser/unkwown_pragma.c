// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

int main(void) {
  #pragma oss taskwhat // expected-error {{unknown OmpSs-2 directive}}
}

