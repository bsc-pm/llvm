// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s


void foo(int *p) {
    struct S *s; // expected-note 2 {{forward declaration of 'struct S'}}
    #pragma oss task depend(in : [4]p[3]) // expected-error {{shaping expression base is not a pointer or array}}
    {}
    #pragma oss task depend(in : [4]s) // expected-error {{shape of pointer to incomplete type 'struct S'}} expected-error {{array has incomplete element type 'struct S'}}
    {}
}
