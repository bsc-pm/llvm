// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -Wuninitialized
static void foo(int *p);
static void foo(int *p); // expected-note {{'foo' declared here}}

int main() {
    int n;
    foo(&n);
    #pragma oss taskwait
}

#pragma oss task inout(*p)
static void foo(int *p) { } // expected-warning {{function has already been declared earlier as a regular (non-task) function, any calls prior to this point will not create tasks}}
