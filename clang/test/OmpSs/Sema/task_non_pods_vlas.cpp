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
  #pragma oss task shared(a, b, c, d, e, f) // expected-error {{Non-POD structs are not supported}}
  { a = *b = c[0] = e[0] = d.x = f.x; }
}

