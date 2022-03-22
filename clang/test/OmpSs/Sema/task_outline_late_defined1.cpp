// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

struct S {
    // expected-note@+1 {{'foo' declared here}}
    void foo();
};

// expected-warning@+2 {{function has already been declared earlier as a regular (non-task) function, any calls prior to this point will not create tasks}}
#pragma oss task
void S::foo() {}

struct P {
    // expected-note@+2 {{'foo' declared here}}
    #pragma oss task
    void foo();
};

// expected-warning@+2 {{function has already been declared earlier as a task function, any calls prior this point may create different tasks}}
#pragma oss task
void P::foo() {}

