// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s
void foo() {
  char cx, ce, cd;

// These additional diagnostics are side effects from the fact of forbidding compare clause
// expected-error@+1 {{unexpected OmpSs-2 clause 'compare' in directive '#pragma oss atomic'}} expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}} expected-note@+2 {{expected built-in binary operator}}
#pragma oss atomic compare
  cx = cx > ce ? ce : cx;
}
