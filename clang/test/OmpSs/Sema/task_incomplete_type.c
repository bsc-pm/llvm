// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

struct S; // expected-note 4 {{forward declaration of 'struct S'}}
extern struct S **p;
extern struct S s;

int main() {
    #pragma oss task in(**p, *p, p[1][2]) // expected-error {{subscript of pointer to incomplete type 'struct S'}} // expected-error {{expression has incomplete type 'struct S'}}
    {}
    #pragma oss task shared(s)
    {}
    #pragma oss task private(s) // expected-error {{expression has incomplete type 'struct S'}}
    {}
    #pragma oss task firstprivate(s) // expected-error {{expression has incomplete type 'struct S'}}
    {}
}

