// RUN: %clang_cc1 -verify -fompss -ferror-limit 100 %s

int main(void) {
  #pragma oss taskwait // expected-error {{oss taskwait not implemented yet}}
}

