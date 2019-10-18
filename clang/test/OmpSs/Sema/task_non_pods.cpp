// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -S -o /dev/null

struct D {
  int x;
  virtual void foo();
};

struct F {
  int x;
};

int main(int argc, char *argv[]) {
  D d;
  F f;
  #pragma oss task in(d, f) // expected-error {{Non-POD structs are not supported}}
  {}
  #pragma oss task shared(d, f) // expected-error {{Non-POD structs are not supported}}
  {}
}

