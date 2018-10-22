// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm

int main(void) {
  #pragma oss taskwait // expected-error {{oss taskwait codegen not supported yet}}
}

