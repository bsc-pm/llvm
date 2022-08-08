// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

template<typename T>
void bar(T p) {
    T t;
    #pragma oss taskiter
    while (t < 10) {} // expected-error {{comparison between pointer and integer ('char *' and 'int')}}
    #pragma oss taskiter
    while (t < p) {}
}

void foo() {
    bar<int>(0);
    bar<char *>(0); // expected-note {{in instantiation of function template specialization 'bar<char *>' requested here}}

    int i;
    #pragma oss taskiter
    while (i < 10) {}
    int *p, *q;
    #pragma oss taskiter
    while (p < q) {}

    #pragma oss taskiter
    while (i++ < 10) {
      for (int j = 0; j < 10; ++j) {
        #pragma oss task firstprivate(j)
        {}
      }
    }
}

