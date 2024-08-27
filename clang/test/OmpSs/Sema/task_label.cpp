// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

template<typename T>
void tfoo(const T *t = nullptr) {
    #pragma oss task label(t) // expected-error {{cannot initialize a variable of type 'const char *' with an lvalue of type 'const int *'}} expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    {}
}

template<typename T>
void tfoo1(const T *t = nullptr) {
    #pragma oss task label("L2", t)
    {}
}

template<int N>
void tbar() {
  int argc;
  char *a;
  #pragma oss task label((N == 1 ? argc : 1) ? "a" : "b") // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
  {}
}

const char *const s = "asdf";
#pragma oss task label(s)
void foo1(); // expected-note {{candidate function}}

#pragma oss task label(p) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
void foo1(const char *p); // expected-note {{candidate function}}

void bar() {
    static const char *const o = "blabla";
    const int *pint = 0;
    #pragma oss task label(o)
    {}
    #pragma oss task label("L1")
    {}
    #pragma oss task label(1) // expected-error{{cannot initialize a variable of type 'const char *' with an rvalue of type 'int'}}
    {}
    #pragma oss task label(pint) // expected-error{{cannot initialize a variable of type 'const char *' with an lvalue of type 'const int *'}}
    {}
    #pragma oss task label(0 ? "a" : "b") // constant evaluated ConditionalOperator condition are valid
    {}
    tfoo<int>(); // expected-note {{in instantiation of function template specialization 'tfoo<int>' requested here}}
    tfoo<char>(); // expected-note {{in instantiation of function template specialization 'tfoo<char>' requested here}}
    tbar<0>();
    tbar<1>(); // expected-note {{in instantiation of function template specialization 'tbar<1>' requested here}}

    // overload
    #pragma oss task label(foo1) // expected-error {{address of overloaded function 'foo1' does not match required type 'char'}}
    {}

    #pragma oss task label(o, // expected-error {{expected ')'}} expected-error {{expected expression}} expected-note {{to match this '('}}
    {}
    const char *a;
    #pragma oss task label(o, a)
    {}
    tfoo<char>();
    #pragma oss task label(false ? a : a) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    {}
    #pragma oss task label(true ? a : a) // expected-error {{C-style string literal, global 'const char *const' initialized variable or a constant ConditionalOperator expression resulting in  a C-style string literal or a global 'const char *const' initialized variable}}
    {}
}
