// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T>
void bar(T &x, T &y) { x.a += y.a; }

namespace N1
{
  struct A { int a; };
  #pragma oss declare reduction(+: A : bar(omp_out, omp_in))
};

#pragma oss declare reduction(+ : int, char : omp_out *= omp_in)
// CHECK: #pragma oss declare reduction (+ : int : omp_out *= omp_in){{$}}
// CHECK-NEXT: #pragma oss declare reduction (+ : char : omp_out *= omp_in)

template <class T>
class SSS {
public:
#pragma oss declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
#pragma oss declare reduction(fun1 : T : omp_out=1, omp_out=foo(omp_in)) initializer(omp_priv = omp_orig + 14)
  static T foo(T &);
};

// CHECK: template <class T> class SSS {
// CHECK: #pragma oss declare reduction (fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
// CHECK: #pragma oss declare reduction (fun1 : T : omp_out = 1 , omp_out = foo(omp_in)) initializer(omp_priv = omp_orig + 14)
// CHECK: };
// CHECK: template<> class SSS<int> {
// CHECK: #pragma oss declare reduction (fun : int : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
// CHECK: #pragma oss declare reduction (fun1 : int : omp_out = 1 , omp_out = foo(omp_in)) initializer(omp_priv = omp_orig + 14)
// CHECK: };

SSS<int> d;

void init(SSS<int> &lhs, SSS<int> rhs);

#pragma oss declare reduction(fun : SSS < int > : omp_out = omp_in) initializer(init(omp_priv, omp_orig))
// CHECK: #pragma oss declare reduction (fun : SSS<int> : omp_out = omp_in) initializer(init(omp_priv, omp_orig))

// CHECK: template <typename T> T foo(T a) {
// CHECK: #pragma oss declare reduction (fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: {
// CHECK: #pragma oss declare reduction (fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: }
// CHECK: return a;
// CHECK: }

// CHECK: template<> int foo<int>(int a) {
// CHECK: #pragma oss declare reduction (fun : int : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: {
// CHECK: #pragma oss declare reduction (fun : int : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: }
// CHECK: return a;
// CHECK: }
template <typename T>
T foo(T a) {
#pragma oss declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
  {
#pragma oss declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
  }
  return a;
}

int main() {
  int i = 0;
  SSS<int> sss;
  // TODO: Add support for scoped reduction identifiers
  //  #pragma oss task reduction(SSS<int>::fun : i)
  // TODO-CHECK: #pragma oss task reduction(SSS<int>::fun: i)
  {
    i += 1;
  }
  // #pragma oss task reduction(::fun:sss)
  // TODO-CHECK: #pragma oss task reduction(::fun: sss)
  {
  }
  N1::A a;
  // CHECK: #pragma oss task reduction(+: a)
  #pragma oss task reduction(+: a)
  {
  }
  return foo(15);
}

#endif
