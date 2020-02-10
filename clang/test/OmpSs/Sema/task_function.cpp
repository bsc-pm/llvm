// RUN: %clang_cc1 -verify -x c++ -fompss-2 -ferror-limit 100 -o - %s

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type&& move(T&& t);
}
}

struct S {
    S(int x);
    ~S();
};
#pragma oss task
void foo1(S&& s) {}
#pragma oss task
void foo2(S const& s) {}
#pragma oss task
void foo3(S& s) {}
// expected-error@+2 {{non-PODs by value are not allowed in task outline}}
#pragma oss task
void foo4(S s) {}

int main() {
    S s(1);
    S&& s1 = S(1);
    S const& s2 = s;
    S& s3 = s;

    foo1(std::move(s1));
    foo1(S(2)); // expected-error {{r-values are not allowed in task outline}}

    foo2(s2);
    foo2(std::move(s2));
    foo2(S(3)); // expected-error {{r-values are not allowed in task outline}}

    foo3(s3);
}
