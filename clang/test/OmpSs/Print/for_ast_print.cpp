// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct S {
  S(): a(0) {}
  S(int v) : a(v) {}
  int a;
  typedef int type;
};

template <typename T>
class S7 : public T {
protected:
  T a;
  T &b;
  typename T::type c:12;
  typename T::type &d;
  S7() : a(0), b(a), c(0), d(a.a) {}

public:
  S7(typename T::type v) : a(v), b(a), c(v), d(a.a) {
#pragma oss task for
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma oss task for
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma oss task for
// CHECK: #pragma oss task for
// CHECK: #pragma oss task for

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma oss task for
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma oss task for
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma oss task for
// CHECK: #pragma oss task for

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  T arr[N];
  static T a;
// CHECK: static T a;
#pragma oss task for
  // CHECK-NEXT: #pragma oss task for
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
  return T();
}

int main(int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  float arr[20];
  static int a;
// CHECK: static int a;
#pragma oss task for
  // CHECK-NEXT: #pragma oss task for
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma oss task for
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      for (auto x : arr)
        foo(), (void)x;
  // CHECK: #pragma oss task for
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: for (auto x : arr)
  // CHECK-NEXT: foo() , (void)x;
  char buf[9] = "01234567";
  char *p, *q;
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
