// RUN: %clang_cc1 -verify -fompss -ferror-limit 100 %s

int main(void) {
  #pragma oss taskwhat // expected-error {{expected an OmpSs directive}}
}

