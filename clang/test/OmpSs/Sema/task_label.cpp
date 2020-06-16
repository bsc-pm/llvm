// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

template<typename T>
void tfoo(const T *t = nullptr) {
    #pragma oss task label(t) // expected-error {{cannot initialize a variable of type 'const char *' with an rvalue of type 'const int *'}}
    {}
}

const char *s = "asdf";
#pragma oss task label(s)
void foo1();

#pragma oss task label(p)
void foo1(const char *p);

void bar() {
    const char *o = "blabla";
    const int *pint = 0;
    #pragma oss task label(o)
    {}
    #pragma oss task label("L1")
    {}
    #pragma oss task label(pint) // expected-error{{cannot initialize a variable of type 'const char *' with an rvalue of type 'const int *'}}
    {}
    tfoo<int>(); // expected-note {{in instantiation of function template specialization 'tfoo<int>' requested here}}
    tfoo<char>();
}
