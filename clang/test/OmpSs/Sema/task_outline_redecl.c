// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s
#pragma oss task inout(*p)
static void foo1(int *p);
static void foo1(int *p); // expected-note {{'foo1' declared here}}
#pragma oss task inout(*p)
static void foo2(int *p); // expected-note {{'foo2' declared here}}
static void foo3(int *p); // expected-note {{'foo3' declared here}}

#pragma oss task inout(*p)
static void foo1(int *p) {} // expected-warning {{function has already been declared earlier as a regular (non-task) function, any calls prior to this point will not create tasks}}
#pragma oss task inout(*p)
static void foo2(int *p) {} // expected-warning {{function has already been declared earlier as a task function, any calls prior this point may create different tasks}}
#pragma oss task inout(*p)
static void foo3(int *p) {} // expected-warning {{function has already been declared earlier as a regular (non-task) function, any calls prior to this point will not create tasks}}

