// RUN: clang-repl -Xcc -E
// RUN: clang-repl -Xcc -emit-llvm 
// expected-no-diagnostics
// We mark it as XFAIL because in upstream the test is failing
// XFAIL: *
