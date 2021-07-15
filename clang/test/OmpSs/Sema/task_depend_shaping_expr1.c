// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

void foo() {

{
    int p[10][20][3];
    int *p1;
    #pragma oss task in([12](p[:])) // expected-error {{OmpSs-2 array section is not allowed here}}
    { }
    #pragma oss task in([[1]p]p) // expected-error {{OmpSs-2 array shaping is not allowed here}}
    { }
    #pragma oss task in([([1]p1)[:]]p)  // expected-error {{OmpSs-2 array shaping is not allowed here}}
    { }
    #pragma oss task in([(p[0:1])]p) // expected-error {{OmpSs-2 array section is not allowed here}}
    { }
    #pragma oss task in([]p) // expected-error {{expected expression}}
    { }
}

{
    int **p;
    #pragma oss task in((p[0:1])[:]) // expected-error {{pointer types only allow one-level array sections}}
    { }
}

}
