// RUN: %clang_cc1 -verify -fompss -ferror-limit 100 %s

int main(void) {
  #pragma oss task // expected-error {{oss task not implemented yet}}
}

