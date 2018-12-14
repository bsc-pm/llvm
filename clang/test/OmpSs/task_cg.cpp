// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -S -o /dev/null

struct D {
  int x;
  virtual void foo();
};

struct F {
  int x;
};

int main(int argc, char *argv[]) {
  int a;
  int *b;
  int c[5];
  int e[argc];
  D d;
  F f;
  #pragma oss task shared(a, b, c, d, e, f) default(none) // expected-error {{PODs are not supported}} expected-error {{VLAs are not supported}}
  {}
}

