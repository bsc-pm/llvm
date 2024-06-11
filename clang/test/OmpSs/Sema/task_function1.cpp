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
// expected-error@+2 {{non-void tasks are not supported except for oss_coroutine return type}}
#pragma oss task
S kk1();

namespace v1 {

template <class Promise = void>
struct coroutine_handle {
};

struct oss_coroutine {
  coroutine_handle<> handle;
};

// expected-error@+2 {{task outline coroutines must define a coroutine}}
#pragma oss task
oss_coroutine kk2();

} // namespace v1

namespace v2 {

struct oss_coroutine {
};

// expected-error@+2 {{oss_coroutine must have a coroutine_handle field}}
#pragma oss task
v2::oss_coroutine kk2();

} // namespace v2

namespace v3 {

template <class Promise = void>
struct coroutine_handle {
};

struct oss_coroutine {
  coroutine_handle<> handle;
};

// expected-error@+2 {{task outline coroutines must define a coroutine}}
#pragma oss task
oss_coroutine kk2() {
  return {};
}

} // namespace v3

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
