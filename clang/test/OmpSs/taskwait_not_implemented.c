// RUN: %clang_cc1 -verify -fompss -ferror-limit 100 %s -S -emit-llvm

int main(void) {
  #pragma oss taskwait // expected-error {{oss taskwait codegen not supported yet}}
}

