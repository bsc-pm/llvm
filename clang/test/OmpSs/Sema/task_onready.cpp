// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

void foo();
void bar() {
    // The runtimes does not have support yet
    #pragma oss task for onready(foo()) // expected-error {{unexpected OmpSs-2 clause 'onready' in directive '#pragma oss task for'}}
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
    {}
    // The runtimes does not have support yet
    #pragma oss taskloop for onready(foo()) // expected-error {{unexpected OmpSs-2 clause 'onready' in directive '#pragma oss taskloop for'}}
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
    {}
}
