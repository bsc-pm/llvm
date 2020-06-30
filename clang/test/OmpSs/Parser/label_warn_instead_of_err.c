// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

// Do not throw an error if label clause looks like mercurium label clause
// NOTE: doing this to be able to run nanos6 tests but keeping clang behaviour
void foo() {
    #pragma oss task label(asdf) // expected-warning {{expecting an expression convertible to 'const char *', ignoring}}
    {}
}
