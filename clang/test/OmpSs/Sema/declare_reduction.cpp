// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
// expected-no-diagnostics

// UNSUPPORTED: true

// We want to check copy constructor.
#pragma oss declare reduction(direct: int: omp_out) initializer(omp_priv(0))

template <typename T>
struct S {
    int x;
    static void bar(T y) {}
    // We want to check that combiner is instantiated although
    // declare reduction type is not dependent
    #pragma oss declare reduction(aaa: int : bar(omp_out)) initializer(omp_priv = 0)
    void foo(T y) {
        #pragma oss task reduction(aaa: y)
        {}
    }
};

int main() {
    S<int> s;
    s.foo(3);
}
