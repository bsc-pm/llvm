// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - %s

struct S {
    // expected-error@+2 {{'#pragma oss task' can only be applied to functions}}
    #pragma oss task in(*p)
    virtual void foo(int *p) {}
};

struct P : S {
    P() {};
    // expected-error@+2 {{'#pragma oss task' can only be applied to functions}}
    #pragma oss task in(*p)
    P(int *p) {}
    // expected-error@+2 {{'#pragma oss task' can only be applied to functions}}
    #pragma oss task
    ~P() {}
    // expected-error@+2 {{'#pragma oss task' can only be applied to functions}}
    #pragma oss task
    P& operator=(const P &) {}
    // expected-error@+2 {{'#pragma oss task' can only be applied to functions}}
    #pragma oss task in(*p)
    void foo(int *p) {}
    #pragma oss task
    void bar() {
      #pragma oss task
      {}
    }

};

#pragma oss task // expected-error {{single declaration is expected after 'task' directive}}
int a, b;
#pragma oss task // expected-error {{single declaration is expected after 'task' directive}}
void foo(), bar();

// expected-error@+2 {{non-void tasks are not supported}}
#pragma oss task
int kk() {}

struct Q {
    int *x;
};

// expected-error@+1 4 {{expected lvalue reference, global variable, dereference, array element, array section or array shape}}
#pragma oss task in((*s).x, *(s->x), &*p, s->x, a, a++, p, *p, p[0 : 4], [10]p)
void foo(Q *s, int a, int *p){}
// expected-error@+1 2 {{expected variable name or array shaping}} expected-error@+1 {{expected lvalue reference or array shape}}
#pragma oss task reduction(+: (*s).x, &*p, a)
void foo_1(Q *s, int a, int *p){}
#pragma oss task in(a)
void foo1(int &a){}
