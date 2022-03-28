// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s

namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type&& move(T&& t);
}
}

struct S {
    #pragma oss task
    void foo1(int x) {}
};

S asdf() {
    return S();
}

void bar() {
    S s;
    S &rs = s;
    std::move(rs).foo1(4);
    asdf().foo1(4 + 5); // expected-warning {{the temporary 'S' in this task call will be destroyed at the end of the expression}}
}
