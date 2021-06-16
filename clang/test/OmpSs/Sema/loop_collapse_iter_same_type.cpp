// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

template<typename T, typename Q>
void foo() {
    Q j; // expected-error {{expected loop iterator type to be 'char' instead of 'int'}}
    T k; // expected-error {{expected loop iterator type to be 'int' instead of 'char'}}
    #pragma oss taskloop collapse(2)
    for (T i = 0; i < 1; ++i) {
      for (j = 0; j < 1; ++j) {
      }
    }
    #pragma oss taskloop collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (k = 0; k < 1; ++k) {
        }
    }
}

typedef int A;
int main() {
    long int j; // expected-error {{expected loop iterator type to be 'int' instead of 'long'}}
    A k;
    #pragma oss taskloop collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (j = 0; j < 1; ++j) {
        }
    }
    #pragma oss taskloop collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (k = 0; k < 1; ++k) {
        }
    }
    foo<char, int>(); // expected-note {{in instantiation of function template specialization 'foo<char, int>' requested here}}
}
