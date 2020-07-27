// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

int main() {
    int v[10][20];
    int v1[10];
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
    #pragma oss task in( { v1[i], i=0::}, v) // expected-error 2 {{expected expression}}
    #pragma oss task in( { v1[i], i=0:10:}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0:10:1}, v)
    #pragma oss task in( { v1[i], i=0:10:1,}, v) // expected-error {{expected iterator identifier}}
    #pragma oss task in( { v[i][j], i=0:10:1,}, v) // expected-error {{expected iterator identifier}} expected-error {{use of undeclared identifier 'j'}}
    #pragma oss task in( { , i=0:10:1}, v) // expected-error {{expected expression}}
    #pragma oss task in( { v1[i], i=0¿10}, v) // expected-error {{expected ':' or ';'}}
    #pragma oss task in( { v1[i], i=0¿10:}, v) // expected-error {{expected ':' or ';'}}

    #pragma oss task in( { v1[i], i = {0, v, 2}} ) // expected-warning {{incompatible pointer to integer conversion initializing 'int' with an expression of type 'int [10][20]'}}
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
    #pragma oss task in( { v[i], i = 0:v:v1 } ) // expected-error {{expression must have integral or unscoped enumeration type, not 'int [10][20]'}} expected-error {{expression must have integral or unscoped enumeration type, not 'int [10]'}}
    {}
}
