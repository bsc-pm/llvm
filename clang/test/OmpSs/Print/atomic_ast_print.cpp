// RUN: %clang_cc1 -verify -fompss-2 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <class T>
T foo(T argc) {
  T v = T();
  T c = T();
  T b = T();
  T a = T();
#pragma oss atomic
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic write
  a = argc + argc;
#pragma oss atomic update
  a = a + argc;
#pragma oss atomic capture
  a = b++;
#pragma oss atomic capture
  {
    a = b;
    b++;
  }
#pragma oss atomic seq_cst
  a++;
#pragma oss atomic read seq_cst
  a = argc;
#pragma oss atomic seq_cst write
  a = argc + argc;
#pragma oss atomic update seq_cst
  a = a + argc;
#pragma oss atomic seq_cst capture
  a = b++;
#pragma oss atomic capture seq_cst
  {
    a = b;
    b++;
  }
#pragma oss atomic
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic write
  a = argc + argc;
#pragma oss atomic update
  a = a + argc;
#pragma oss atomic acq_rel capture
  a = b++;
#pragma oss atomic capture acq_rel
  {
    a = b;
    b++;
  }
#pragma oss atomic
  a++;
#pragma oss atomic read acquire
  a = argc;
#pragma oss atomic write
  a = argc + argc;
#pragma oss atomic update
  a = a + argc;
#pragma oss atomic acquire capture
  a = b++;
#pragma oss atomic capture acquire
  {
    a = b;
    b++;
  }
#pragma oss atomic release
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic release write
  a = argc + argc;
#pragma oss atomic update release
  a = a + argc;
#pragma oss atomic release capture
  a = b++;
#pragma oss atomic capture release
  {
    a = b;
    b++;
  }
#pragma oss atomic relaxed
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic relaxed write
  a = argc + argc;
#pragma oss atomic update relaxed
  a = a + argc;
#pragma oss atomic relaxed capture
  a = b++;
#pragma oss atomic capture relaxed
  {
    a = b;
    b++;
  }
  return T();
}

// CHECK: T a = T();
// CHECK-NEXT: #pragma oss atomic{{$}}
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic seq_cst
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read seq_cst
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic seq_cst write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update seq_cst
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic seq_cst capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture seq_cst
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read acquire
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic acquire capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture acquire
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic release
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic release write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update release
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic release capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture release
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic relaxed
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic relaxed write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update relaxed
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic relaxed capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture relaxed
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK: int a = int();
// CHECK-NEXT: #pragma oss atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic seq_cst
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read seq_cst
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic seq_cst write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update seq_cst
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic seq_cst capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture seq_cst
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read acquire
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic acquire capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture acquire
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic release
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic release write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update release
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic release capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture release
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma oss atomic relaxed
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma oss atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma oss atomic relaxed write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma oss atomic update relaxed
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma oss atomic relaxed capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma oss atomic capture relaxed
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }

int main(int argc, char **argv) {
  int v = 0;
  int c = 0;
  int b = 0;
  int a = 0;
// CHECK: int a = 0;
#pragma oss atomic
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic write
  a = argc + argc;
#pragma oss atomic update
  a = a + argc;
#pragma oss atomic capture
  a = b++;
#pragma oss atomic capture
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma oss atomic compare
  { a = a > b ? b : a; }
#pragma oss atomic compare
  { a = a < b ? b : a; }
#pragma oss atomic compare
  { a = a == b ? c : a; }
#pragma oss atomic compare capture
  { v = a; if (a > b) { a = b; } }
#pragma oss atomic compare capture
  { v = a; if (a < b) { a = b; } }
#pragma oss atomic compare capture
  { v = a == b; if (v) a = c; }
#endif
#pragma oss atomic seq_cst
  a++;
#pragma oss atomic read seq_cst
  a = argc;
#pragma oss atomic seq_cst write
  a = argc + argc;
#pragma oss atomic update seq_cst
  a = a + argc;
#pragma oss atomic seq_cst capture
  a = b++;
#pragma oss atomic capture seq_cst
  {
    a = b;
    b++;
  }
#pragma oss atomic
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic write
  a = argc + argc;
#pragma oss atomic update
  a = a + argc;
#pragma oss atomic acq_rel capture
  a = b++;
#pragma oss atomic capture acq_rel
  {
    a = b;
    b++;
  }
#pragma oss atomic
  a++;
#pragma oss atomic read acquire
  a = argc;
#pragma oss atomic write
  a = argc + argc;
#pragma oss atomic update
  a = a + argc;
#pragma oss atomic acquire capture
  a = b++;
#pragma oss atomic capture acquire
  {
    a = b;
    b++;
  }
#pragma oss atomic release
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic release write
  a = argc + argc;
#pragma oss atomic update release
  a = a + argc;
#pragma oss atomic release capture
  a = b++;
#pragma oss atomic capture release
  {
    a = b;
    b++;
  }
#pragma oss atomic relaxed
  a++;
#pragma oss atomic read
  a = argc;
#pragma oss atomic relaxed write
  a = argc + argc;
#pragma oss atomic update relaxed
  a = a + argc;
#pragma oss atomic relaxed capture
  a = b++;
#pragma oss atomic capture relaxed
  {
    a = b;
    b++;
  }
  // CHECK-NEXT: #pragma oss atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma oss atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma oss atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma oss atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma oss atomic capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma oss atomic capture
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma oss atomic seq_cst
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma oss atomic read seq_cst
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma oss atomic seq_cst write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma oss atomic update seq_cst
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma oss atomic seq_cst capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma oss atomic capture seq_cst
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma oss atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma oss atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma oss atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma oss atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma oss atomic acq_rel capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma oss atomic capture acq_rel
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma oss atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma oss atomic read acquire
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma oss atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma oss atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma oss atomic acquire capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma oss atomic capture acquire
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma oss atomic release
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma oss atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma oss atomic release write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma oss atomic update release
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma oss atomic release capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma oss atomic capture release
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma oss atomic relaxed
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma oss atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma oss atomic relaxed write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma oss atomic update relaxed
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma oss atomic relaxed capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma oss atomic capture relaxed
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // expect-note@+1 {{in instantiation of function template specialization 'foo<int>' requested here}}
  return foo(a);
}

#ifdef OMP51

template <typename Ty> Ty ffoo(Ty *x, Ty e, Ty d) {
  Ty v;
  bool r;

#pragma oss atomic compare capture
  {
    v = *x;
    if (*x > e)
      *x = e;
  }
#pragma oss atomic compare capture
  {
    v = *x;
    if (*x < e)
      *x = e;
  }
#pragma oss atomic compare capture
  {
    v = *x;
    if (*x == e)
      *x = d;
  }
#pragma oss atomic compare capture
  {
    if (*x > e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture
  {
    if (*x < e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture
  {
    if (*x == e)
      *x = d;
    v = *x;
  }
#pragma oss atomic compare capture
  {
    if (*x == e)
      *x = d;
    else
      v = *x;
  }
#pragma oss atomic compare capture
  {
    r = *x == e;
    if (r)
      *x = d;
  }
#pragma oss atomic compare capture
  {
    r = *x == e;
    if (r)
      *x = d;
    else
      v = *x;
  }

#pragma oss atomic compare capture acq_rel
  {
    v = *x;
    if (*x > e)
      *x = e;
  }
#pragma oss atomic compare capture acq_rel
  {
    v = *x;
    if (*x < e)
      *x = e;
  }
#pragma oss atomic compare capture acq_rel
  {
    v = *x;
    if (*x == e)
      *x = d;
  }
#pragma oss atomic compare capture acq_rel
  {
    if (*x > e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture acq_rel
  {
    if (*x < e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture acq_rel
  {
    if (*x == e)
      *x = d;
    v = *x;
  }
#pragma oss atomic compare capture acq_rel
  {
    if (*x == e)
      *x = d;
    else
      v = *x;
  }
#pragma oss atomic compare capture acq_rel
  {
    r = *x == e;
    if (r)
      *x = d;
  }
#pragma oss atomic compare capture acq_rel
  {
    r = *x == e;
    if (r)
      *x = d;
    else
      v = *x;
  }

#pragma oss atomic compare capture acquire
  {
    v = *x;
    if (*x > e)
      *x = e;
  }
#pragma oss atomic compare capture acquire
  {
    v = *x;
    if (*x < e)
      *x = e;
  }
#pragma oss atomic compare capture acquire
  {
    v = *x;
    if (*x == e)
      *x = d;
  }
#pragma oss atomic compare capture acquire
  {
    if (*x > e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture acquire
  {
    if (*x < e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture acquire
  {
    if (*x == e)
      *x = d;
    v = *x;
  }
#pragma oss atomic compare capture acquire
  {
    if (*x == e)
      *x = d;
    else
      v = *x;
  }
#pragma oss atomic compare capture acquire
  {
    r = *x == e;
    if (r)
      *x = d;
  }
#pragma oss atomic compare capture acquire
  {
    r = *x == e;
    if (r)
      *x = d;
    else
      v = *x;
  }

#pragma oss atomic compare capture relaxed
  {
    v = *x;
    if (*x > e)
      *x = e;
  }
#pragma oss atomic compare capture relaxed
  {
    v = *x;
    if (*x < e)
      *x = e;
  }
#pragma oss atomic compare capture relaxed
  {
    v = *x;
    if (*x == e)
      *x = d;
  }
#pragma oss atomic compare capture relaxed
  {
    if (*x > e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture relaxed
  {
    if (*x < e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture relaxed
  {
    if (*x == e)
      *x = d;
    v = *x;
  }
#pragma oss atomic compare capture relaxed
  {
    if (*x == e)
      *x = d;
    else
      v = *x;
  }
#pragma oss atomic compare capture relaxed
  {
    r = *x == e;
    if (r)
      *x = d;
  }
#pragma oss atomic compare capture relaxed
  {
    r = *x == e;
    if (r)
      *x = d;
    else
      v = *x;
  }

#pragma oss atomic compare capture release
  {
    v = *x;
    if (*x > e)
      *x = e;
  }
#pragma oss atomic compare capture release
  {
    v = *x;
    if (*x < e)
      *x = e;
  }
#pragma oss atomic compare capture release
  {
    v = *x;
    if (*x == e)
      *x = d;
  }
#pragma oss atomic compare capture release
  {
    if (*x > e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture release
  {
    if (*x < e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture release
  {
    if (*x == e)
      *x = d;
    v = *x;
  }
#pragma oss atomic compare capture release
  {
    if (*x == e)
      *x = d;
    else
      v = *x;
  }
#pragma oss atomic compare capture release
  {
    r = *x == e;
    if (r)
      *x = d;
  }
#pragma oss atomic compare capture release
  {
    r = *x == e;
    if (r)
      *x = d;
    else
      v = *x;
  }

#pragma oss atomic compare capture seq_cst
  {
    v = *x;
    if (*x > e)
      *x = e;
  }
#pragma oss atomic compare capture seq_cst
  {
    v = *x;
    if (*x < e)
      *x = e;
  }
#pragma oss atomic compare capture seq_cst
  {
    v = *x;
    if (*x == e)
      *x = d;
  }
#pragma oss atomic compare capture seq_cst
  {
    if (*x > e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture seq_cst
  {
    if (*x < e)
      *x = e;
    v = *x;
  }
#pragma oss atomic compare capture seq_cst
  {
    if (*x == e)
      *x = d;
    v = *x;
  }
#pragma oss atomic compare capture seq_cst
  {
    if (*x == e)
      *x = d;
    else
      v = *x;
  }
#pragma oss atomic compare capture seq_cst
  {
    r = *x == e;
    if (r)
      *x = d;
  }
#pragma oss atomic compare capture seq_cst
  {
    r = *x == e;
    if (r)
      *x = d;
    else
      v = *x;
  }

  return v;
}

void bbaarr() {
  {
    char x, e, d;
    ffoo(&x, e, d);
  }

  {
    unsigned char x, e, d;
    ffoo(&x, e, d);
  }

  {
    short x, e, d;
    ffoo(&x, e, d);
  }

  {
    unsigned short x, e, d;
    ffoo(&x, e, d);
  }

  {
    int x, e, d;
    ffoo(&x, e, d);
  }

  {
    unsigned int x, e, d;
    ffoo(&x, e, d);
  }

  {
    long x, e, d;
    ffoo(&x, e, d);
  }

  {
    unsigned long x, e, d;
    ffoo(&x, e, d);
  }

  {
    long long x, e, d;
    ffoo(&x, e, d);
  }

  {
    unsigned long long x, e, d;
    ffoo(&x, e, d);
  }

  {
    float x, e, d;
    ffoo(&x, e, d);
  }

  {
    double x, e, d;
    ffoo(&x, e, d);
  }
}

// CHECK-51: template <typename Ty> Ty ffoo(Ty *x, Ty e, Ty d) {

int xevd() {
  int x, v, e, d;

#pragma oss atomic compare capture
  {
    v = x;
    x = x > e ? e : x;
  }
#pragma oss atomic compare capture
  {
    v = x;
    x = x < e ? e : x;
  }
#pragma oss atomic compare capture
  {
    v = x;
    x = x == e ? d : x;
  }
#pragma oss atomic compare capture
  {
    x = x > e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture
  {
    x = x < e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture
  {
    x = x == e ? d : x;
    v = x;
  }

#pragma oss atomic compare capture acq_rel
  {
    v = x;
    x = x > e ? e : x;
  }
#pragma oss atomic compare capture acq_rel
  {
    v = x;
    x = x < e ? e : x;
  }
#pragma oss atomic compare capture acq_rel
  {
    v = x;
    x = x == e ? d : x;
  }
#pragma oss atomic compare capture acq_rel
  {
    x = x > e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture acq_rel
  {
    x = x < e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture acq_rel
  {
    x = x == e ? d : x;
    v = x;
  }

#pragma oss atomic compare capture acquire
  {
    v = x;
    x = x > e ? e : x;
  }
#pragma oss atomic compare capture acquire
  {
    v = x;
    x = x < e ? e : x;
  }
#pragma oss atomic compare capture acquire
  {
    v = x;
    x = x == e ? d : x;
  }
#pragma oss atomic compare capture acquire
  {
    x = x > e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture acquire
  {
    x = x < e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture acquire
  {
    x = x == e ? d : x;
    v = x;
  }

#pragma oss atomic compare capture relaxed
  {
    v = x;
    x = x > e ? e : x;
  }
#pragma oss atomic compare capture relaxed
  {
    v = x;
    x = x < e ? e : x;
  }
#pragma oss atomic compare capture relaxed
  {
    v = x;
    x = x == e ? d : x;
  }
#pragma oss atomic compare capture relaxed
  {
    x = x > e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture relaxed
  {
    x = x < e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture relaxed
  {
    x = x == e ? d : x;
    v = x;
  }

#pragma oss atomic compare capture release
  {
    v = x;
    x = x > e ? e : x;
  }
#pragma oss atomic compare capture release
  {
    v = x;
    x = x < e ? e : x;
  }
#pragma oss atomic compare capture release
  {
    v = x;
    x = x == e ? d : x;
  }
#pragma oss atomic compare capture release
  {
    x = x > e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture release
  {
    x = x < e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture release
  {
    x = x == e ? d : x;
    v = x;
  }

#pragma oss atomic compare capture seq_cst
  {
    v = x;
    x = x > e ? e : x;
  }
#pragma oss atomic compare capture seq_cst
  {
    v = x;
    x = x < e ? e : x;
  }
#pragma oss atomic compare capture seq_cst
  {
    v = x;
    x = x == e ? d : x;
  }
#pragma oss atomic compare capture seq_cst
  {
    x = x > e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture seq_cst
  {
    x = x < e ? e : x;
    v = x;
  }
#pragma oss atomic compare capture seq_cst
  {
    x = x == e ? d : x;
    v = x;
  }

  return v;
}

#endif

#endif
