// RUN: %clang_cc1 -verify -fompss-2 -x c++ -std=c++11 -fexceptions -fcxx-exceptions %s -Wuninitialized

class S {
  int a;
  S() : a(0) {}

public:
  S(int v) : a(v) {}
  S(const S &s) : a(s.a) {}
};

static int sii;
static int globalii;

// Currently, we cannot use "0" for global register variables.
// register int reg0 __asm__("0");
int reg0;

int test_iteration_spaces() {
  const int N = 100;
  float a[N], b[N], c[N];
  int ii, jj, kk;
  float fii;
  double dii;
#pragma oss task for
  for (int i = 0; i < 10; i += 1) {
    c[i] = a[i] + b[i];
  }
#pragma oss task for
  for (char i = 0; i < 10; i++) {
    c[i] = a[i] + b[i];
  }
#pragma oss task for
  for (char i = 0; i < 10; i += '\1') {
    c[i] = a[i] + b[i];
  }
#pragma oss task for
  for (long long i = 0; i < 10; i++) {
    c[i] = a[i] + b[i];
  }
// expected-error@+2 {{expression must have integral or unscoped enumeration type, not 'double'}}
#pragma oss task for
  for (long long i = 0; i < 10; i += 1.5) {
    c[i] = a[i] + b[i];
  }
#pragma oss task for
  for (long long i = 0; i < 'z'; i += 1u) {
    c[i] = a[i] + b[i];
  }
// expected-error@+2 {{variable must be of integer type}}
#pragma oss task for
  for (float fi = 0; fi < 10.0; fi++) {
    c[(int)fi] = a[(int)fi] + b[(int)fi];
  }
// expected-error@+2 {{variable must be of integer type}}
#pragma oss task for
  for (double fi = 0; fi < 10.0; fi++) {
    c[(int)fi] = a[(int)fi] + b[(int)fi];
  }
// expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (int &ref = ii; ref < 10; ref++) {
  }
// expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (int i; i < 10; i++)
    c[i] = a[i];

// expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (int i = 0, j = 0; i < 10; ++i)
    c[i] = a[i];

// expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (; ii < 10; ++ii)
    c[ii] = a[ii];

// expected-warning@+3 {{expression result unused}}
// expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (ii + 1; ii < 10; ++ii)
    c[ii] = a[ii];

// expected-error@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (c[ii] = 0; ii < 10; ++ii)
    c[ii] = a[ii];

// Ok to skip parenthesises.
#pragma oss task for
  for (((ii)) = 0; ii < 10; ++ii)
    c[ii] = a[ii];

// expected-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', or '>=') of loop variable 'i'}} omp5-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
#pragma oss task for
  for (int i = 0; i; i++)
    c[i] = a[i];

// expected-error@+3 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', or '>=') of loop variable 'i'}} omp5-error@+3 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'i'}}
#pragma oss task for
  for (int i = 0; jj < kk; ii++)
    c[i] = a[i];

// expected-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', or '>=') of loop variable 'i'}} omp5-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
#pragma oss task for
  for (int i = 0; !!i; i++)
    c[i] = a[i];

// expected-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', or '>=') of loop variable 'i'}}
#pragma oss task for
  for (int i = 0; i != 1; i++)
    c[i] = a[i];

// expected-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', or '>=') of loop variable 'i'}} omp5-error@+2 {{condition of OmpSs-2 for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
#pragma oss task for
  for (int i = 0;; i++)
    c[i] = a[i];

// Ok.
#pragma oss task for
  for (int i = 11; i > 10; i--)
    c[i] = a[i];

// Ok.
#pragma oss task for
  for (int i = 0; i < 10; ++i)
    c[i] = a[i];

// Ok.
#pragma oss task for
  for (ii = 0; ii < 10; ++ii)
    c[ii] = a[ii];

// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10; ++jj)
    c[ii] = a[jj];

// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10; ++++ii)
    c[ii] = a[ii];

// Ok but undefined behavior (in general, cannot check that incr
// is really loop-invariant).
#pragma oss task for
  for (ii = 0; ii < 10; ii = ii + ii)
    c[ii] = a[ii];

// expected-error@+2 {{expression must have integral or unscoped enumeration type, not 'float'}}
#pragma oss task for
  for (ii = 0; ii < 10; ii = ii + 1.0f)
    c[ii] = a[ii];

// Ok - step was converted to integer type.
#pragma oss task for
  for (ii = 0; ii < 10; ii = ii + (int)1.1f)
    c[ii] = a[ii];

// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10; jj = ii + 2)
    c[ii] = a[ii];

// expected-warning@+3 {{relational comparison result unused}}
// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii<10; jj> kk + 2)
    c[ii] = a[ii];

// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10;)
    c[ii] = a[ii];

// expected-warning@+3 {{expression result unused}}
// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10; !ii)
    c[ii] = a[ii];

// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10; ii ? ++ii : ++jj)
    c[ii] = a[ii];

// expected-error@+2 {{increment clause of OmpSs-2 for loop must perform simple addition or subtraction on loop variable 'ii'}}
#pragma oss task for
  for (ii = 0; ii < 10; ii = ii < 10)
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; ii < 10; ii = ii + 0)
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; ii < 10; ii = ii + (int)(0.8 - 0.45))
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; (ii) < 10; ii -= 25)
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; (ii < 10); ii -= 0)
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be negative due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to decrease on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; ii > 10; (ii += 0))
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; ii < 10; (ii) = (1 - 1) + (ii))
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be negative due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to decrease on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for ((ii = 0); ii > 10; (ii -= 0))
    c[ii] = a[ii];

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'ii' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (ii = 0; (ii < 10); (ii -= 0))
    c[ii] = a[ii];

// expected-error@+3 {{the loop initializer expression depends on the current loop control variable}}
// expected-error@+2 2 {{the loop condition expression depends on the current loop control variable}}
#pragma oss task for
  for (ii = ii * 10 + 25; ii < ii / ii - 23; ii += 1)
    c[ii] = a[ii];

// expected-error@+2 {{firstprivate variable cannot be private}}
#pragma oss task for firstprivate(ii)
  for (ii = 0; ii < 10; ii++)
    c[ii] = a[ii];

#pragma oss task for private(ii)
  for (ii = 0; ii < 10; ii++)
    c[ii] = a[ii];

  {
#pragma oss task for
    for (reg0 = 0; reg0 < 10; reg0 += 1)
      c[reg0] = a[reg0];
  }

  {
#pragma oss task for
    for (globalii = 0; globalii < 10; globalii += 1)
      c[globalii] = a[globalii];
  }

// expected-error@+2 {{statement after '#pragma oss task for' must be a for loop}}
#pragma oss task for
  for (auto &item : a) {
    item = item + 1;
  }

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'i' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (unsigned i = 9; i < 10; i--) {
    c[i] = a[i] + b[i];
  }

  int(*lb)[4] = nullptr;
// expected-error@+2 {{variable must be of integer type}}
#pragma oss task for
  for (int(*p)[4] = lb; p < lb + 8; ++p) {
  }

// expected-warning@+2 {{initialization clause of OmpSs-2 for loop is not in canonical form ('var = init' or 'T var = init')}}
#pragma oss task for
  for (int a{0}; a < 10; ++a) {
  }

  return 0;
}

template <typename IT, int ST>
class TC {
  int ii, iii, kk;
public:
  enum { myconstant = 42 };
  int ub();
  int dotest_lt(IT begin, IT end) {
// expected-error@+3 3 {{the loop initializer expression depends on the current loop control variable}}
// expected-error@+2 6 {{the loop condition expression depends on the current loop control variable}}
#pragma oss task for
  for (int ii = ii * 10 + 25; ii < ii / ii - 23; ii += 1)
    ;

// Check that member function calls and enum constants in the condition is
// handled.
#pragma oss task for
  for (int ii = 0; ii < ub() + this->myconstant; ii += 1) // expected-no-error
    ;

// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'I' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
    for (IT I = begin; I < end; I = I + ST) {
      ++I;
    }
// expected-note@+3 {{loop step is expected to be positive due to this condition}}
// expected-error@+2 {{increment expression must cause 'I' to increase on each iteration of OmpSs-2 for loop}}
#pragma oss task for
    for (IT I = begin; I <= end; I += ST) {
      ++I;
    }
#pragma oss task for
    for (IT I = begin; I < end; ++I) {
      ++I;
    }
  }

  static IT step() {
    return IT(ST);
  }
};
template <typename IT, int ST = 0>
int dotest_gt(IT begin, IT end) {
// expected-note@+3 {{loop step is expected to be negative due to this condition}}
// expected-error@+2 {{increment expression must cause 'I' to decrease on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (IT I = begin; I >= end; I = I + ST) {
    ++I;
  }
// expected-note@+3 {{loop step is expected to be negative due to this condition}}
// expected-error@+2 {{increment expression must cause 'I' to decrease on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (IT I = begin; I >= end; I += ST) {
    ++I;
  }

// expected-note@+3 {{loop step is expected to be negative due to this condition}}
// expected-error@+2 {{increment expression must cause 'I' to decrease on each iteration of OmpSs-2 for loop}}
#pragma oss task for
  for (IT I = begin; I >= end; ++I) {
    ++I;
  }

#pragma oss task for
  for (IT I = begin; I < end; I += TC<int, ST>::step()) {
    ++I;
  }
}

void test_with_template() {
  TC<int, 100> t1;
  TC<int, -100> t2;
  t1.dotest_lt(0, 100); // expected-note {{in instantiation of member function 'TC<int, 100>::dotest_lt' requested here}}
  t2.dotest_lt(0, 100); // expected-note {{in instantiation of member function 'TC<int, -100>::dotest_lt' requested here}}
  dotest_gt<unsigned, 10>(0, 100);  // expected-note {{in instantiation of function template specialization 'dotest_gt<unsigned int, 10>' requested here}}
}

void test_loop_break() {
  const int N = 100;
  float a[N], b[N], c[N];
#pragma oss task for
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + b[i];
    for (int j = 0; j < 10; ++j) {
      if (a[i] > b[j])
        break; // OK in nested loop
    }
    switch (i) {
    case 1:
      b[i]++;
      break;
    default:
      break;
    }
    if (c[i] > 10)
      break; // expected-error {{'break' statement cannot be used in OmpSs-2 for loop}}

    if (c[i] > 11)
      break; // expected-error {{'break' statement cannot be used in OmpSs-2 for loop}}
  }

#pragma oss task for
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i] = a[i] + b[i];
      if (c[i] > 10) {
        if (c[i] < 20) {
          break; // OK
        }
      }
    }
  }
}

void test_loop_eh() {
  const int N = 100;
  float a[N], b[N], c[N];
#pragma oss task for
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + b[i];
    try {
      for (int j = 0; j < 10; ++j) {
        if (a[i] > b[j])
          throw a[i];
      }
      throw a[i];
    } catch (float f) {
      if (f > 0.1)
        throw a[i];
      return; // expected-error {{invalid branch from OmpSs-2 structured block}}
    }
    switch (i) {
    case 1:
      b[i]++;
      break;
    default:
      break;
    }
    for (int j = 0; j < 10; j++) {
      if (c[i] > 10)
        throw c[i];
    }
  }
  if (c[9] > 10)
    throw c[9]; // OK

#pragma oss task for
  for (int i = 0; i < 10; ++i) {
    struct S {
      void g() { throw 0; }
    };
  }
}

