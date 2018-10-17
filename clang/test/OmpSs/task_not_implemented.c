// RUN: %clang_cc1 -verify -fompss -ferror-limit 100 %s -S -emit-llvm

int main(void) {
  #pragma oss task // expected-error {{oss task codegen not supported yet}}
}

