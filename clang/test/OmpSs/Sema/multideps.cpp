// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
template<typename T>
void foo() {
    T v[10][10];
    T v1[10];
    int i;
    #pragma oss task in( { v1[i] }, v) // expected-error {{expected ','}} expected-error {{expected iterator identifier}}
    #pragma oss task in( { v1[i], }, v) // expected-error {{expected iterator identifier}}
    #pragma oss task in( { v1[i], i}, v) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], k=0;10:1}, v) // this is ok
    #pragma oss task in( { v1[i], i, j}, v) // expected-error 2 {{expected '='}}
    #pragma oss task in( { v1[i], i=}, v) // expected-error {{expected expression}} expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i=0}, v) // expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i=;}, v) // expected-error 2 {{expected expression}}
    #pragma oss task in( { v1[i], i=0;}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0;:}, v) // expected-error 2 {{expected expression}}
    #pragma oss task in( { v1[i], i=0;10:}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0;10:1}, v)
    #pragma oss task in( { v1[i], i=0;10:1,}, v) // expected-error {{expected iterator identifier}}
    #pragma oss task in( { v[i][j], i=0;10:1,}, v) // expected-error {{expected iterator identifier}} expected-error {{use of undeclared identifier 'j'}}
    #pragma oss task in( { , i=0;10:1}, v) // expected-error {{expected expression}}

    #pragma oss task in( { v1[i] }, v) // expected-error {{expected ','}} expected-error {{expected iterator identifier}}
    #pragma oss task in( { v1[i], }, v) // expected-error {{expected iterator identifier}}
    #pragma oss task in( { v1[i], i}, v) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], k=0:10:1}, v) // this is ok
    #pragma oss task in( { v1[i], i, j}, v) // expected-error 2 {{expected '='}}
    #pragma oss task in( { v1[i], i=}, v) // expected-error {{expected expression}} expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i=:}, v) // expected-error 2 {{expected expression}}
    #pragma oss task in( { v1[i], i=0:}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0::}, v) // expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i=0:10:}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0:10:1}, v)
    #pragma oss task in( { v1[i], i=0:10:1,}, v) // expected-error {{expected iterator identifier}}
    #pragma oss task in( { v[i][j], i=0:10:1,}, v) // expected-error {{expected iterator identifier}} expected-error {{use of undeclared identifier 'j'}}
    #pragma oss task in( { , i=0:10:1}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0¿10}, v) // expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i=0¿10:}, v) // expected-error {{expected ':' or ';'}}

    #pragma oss task in( { v1[i], i = {0, v, 2}} ) // expected-error {{cannot initialize an array element of type 'int' with an lvalue of type 'int [10][10]'}}
    #pragma oss task in( { v1[i], i|0:10:0, j = 0 }, v) // expected-error {{expected '='}} expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i|}, v) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], i(1)}, v ) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], i(1):10:1}, v ) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], i(1):10:1, j}, v ) // expected-error 2 {{expected '='}}
    #pragma oss task in( { v1[i], i{1}}, v ) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], i{1}:10:1}, v ) // expected-error {{expected '='}}
    #pragma oss task in( { v1[i], i{1}:10:1, j}, v ) // expected-error 2 {{expected '='}}

    #pragma oss task in( { v1[i], i = 0:10:1 }[2], v ) // expected-error {{expected ',' or ')' in 'in' clause}} expected-error {{expected expression}}
    #pragma oss task in( { v[i], i = { ,}, j = i: } ) // expected-error 2 {{expected expression}}
    #pragma oss task in( { v[i], i = 0:v:v1 } ) // expected-error {{expression must have integral or unscoped enumeration type, not 'int [10][10]'}} expected-error {{expression must have integral or unscoped enumeration type, not 'int [10]'}}
    {}
}

void bar() {
    foo<int>(); // expected-note {{in instantiation of function template specialization 'foo<int>' requested here}}
}

struct S {
    int x;
    S();
    S(int);
    operator int();
};

template<typename T>
void foo1() {
    T v[10];
    // This is ok because it has conversion to 'int'
    #pragma oss task in({ v[i], i=T(0): T(10): T(1) })
    {}
}

void bar1() {
    foo1<S>();
}
