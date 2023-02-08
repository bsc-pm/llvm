// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -Wuninitialized
void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma oss atomic read
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

int foo(void) {
L1:
  foo();
#pragma oss atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
    foo();
    goto L1;
  }
  goto L2;
#pragma oss atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
    foo();
  L2:
    foo();
  }

  return 0;
}

struct S {
  int a;
};

int readint(void) {
  int a = 0, b = 0;
// Test for atomic read
#pragma oss atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma oss atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  foo();
#pragma oss atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  a += b;
#pragma oss atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected lvalue expression}}
  a = 0;
#pragma oss atomic read
  a = b;
  // expected-error@+1 {{directive '#pragma oss atomic' cannot contain more than one 'read' clause}}
#pragma oss atomic read read
  a = b;

  return 0;
}

int readS(void) {
  struct S a, b;
  // expected-error@+1 {{directive '#pragma oss atomic' cannot contain more than one 'read' clause}}
#pragma oss atomic read read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected expression of scalar type}}
  a = b;

  return a.a;
}

int writeint(void) {
  int a = 0, b = 0;
// Test for atomic write
#pragma oss atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma oss atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  foo();
#pragma oss atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  a += b;
#pragma oss atomic write
  a = 0;
#pragma oss atomic write
  a = b;
  // expected-error@+1 {{directive '#pragma oss atomic' cannot contain more than one 'write' clause}}
#pragma oss atomic write write
  a = b;

  return 0;
}

int writeS(void) {
  struct S a, b;
  // expected-error@+1 {{directive '#pragma oss atomic' cannot contain more than one 'write' clause}}
#pragma oss atomic write write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected expression of scalar type}}
  a = b;

  return a.a;
}

int updateint(void) {
  int a = 0, b = 0;
// Test for atomic update
#pragma oss atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma oss atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in binary or unary operator}}
  foo();
#pragma oss atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in binary operator}}
  a = b;
#pragma oss atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  a = b || a;
#pragma oss atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  a = a && b;
#pragma oss atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = (float)a + b;
#pragma oss atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = 2 * b;
#pragma oss atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = b + *&a;
#pragma oss atomic update
  *&a = *&a +  2;
#pragma oss atomic update
  a++;
#pragma oss atomic
  ++a;
#pragma oss atomic update
  a--;
#pragma oss atomic
  --a;
#pragma oss atomic update
  a += b;
#pragma oss atomic
  a %= b;
#pragma oss atomic update
  a *= b;
#pragma oss atomic
  a -= b;
#pragma oss atomic update
  a /= b;
#pragma oss atomic
  a &= b;
#pragma oss atomic update
  a ^= b;
#pragma oss atomic
  a |= b;
#pragma oss atomic update
  a <<= b;
#pragma oss atomic
  a >>= b;
#pragma oss atomic update
  a = b + a;
#pragma oss atomic
  a = a * b;
#pragma oss atomic update
  a = b - a;
#pragma oss atomic
  a = a / b;
#pragma oss atomic update
  a = b & a;
#pragma oss atomic
  a = a ^ b;
#pragma oss atomic update
  a = b | a;
#pragma oss atomic
  a = a << b;
#pragma oss atomic
  a = b >> a;
  // expected-error@+1 {{directive '#pragma oss atomic' cannot contain more than one 'update' clause}}
#pragma oss atomic update update
  a /= b;

  return 0;
}

int captureint(void) {
  int a = 0, b = 0, c = 0;
// Test for atomic capture
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected compound statement}}
  ;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  foo();
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in binary or unary operator}}
  a = b;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = b || a;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  b = a = a && b;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = (float)a + b;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = 2 * b;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = b + *&a;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected exactly two expression statements}}
  { a = b; }
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected exactly two expression statements}}
  {}
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of the first expression}}
  {a = b;a = b;}
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of the first expression}}
  {a = b; a = b || a;}
#pragma oss atomic capture
  {b = a; a = a && b;}
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = (float)a + b;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = 2 * b;
#pragma oss atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = b + *&a;
#pragma oss atomic capture
  c = *&a = *&a +  2;
#pragma oss atomic capture
  c = a++;
#pragma oss atomic capture
  c = ++a;
#pragma oss atomic capture
  c = a--;
#pragma oss atomic capture
  c = --a;
#pragma oss atomic capture
  c = a += b;
#pragma oss atomic capture
  c = a %= b;
#pragma oss atomic capture
  c = a *= b;
#pragma oss atomic capture
  c = a -= b;
#pragma oss atomic capture
  c = a /= b;
#pragma oss atomic capture
  c = a &= b;
#pragma oss atomic capture
  c = a ^= b;
#pragma oss atomic capture
  c = a |= b;
#pragma oss atomic capture
  c = a <<= b;
#pragma oss atomic capture
  c = a >>= b;
#pragma oss atomic capture
  c = a = b + a;
#pragma oss atomic capture
  c = a = a * b;
#pragma oss atomic capture
  c = a = b - a;
#pragma oss atomic capture
  c = a = a / b;
#pragma oss atomic capture
  c = a = b & a;
#pragma oss atomic capture
  c = a = a ^ b;
#pragma oss atomic capture
  c = a = b | a;
#pragma oss atomic capture
  c = a = a << b;
#pragma oss atomic capture
  c = a = b >> a;
#pragma oss atomic capture
  { c = *&a; *&a = *&a +  2;}
#pragma oss atomic capture
  { *&a = *&a +  2; c = *&a;}
#pragma oss atomic capture
  {c = a; a++;}
#pragma oss atomic capture
  {c = a; (a)++;}
#pragma oss atomic capture
  {++a;c = a;}
#pragma oss atomic capture
  {c = a;a--;}
#pragma oss atomic capture
  {--a;c = a;}
#pragma oss atomic capture
  {c = a; a += b;}
#pragma oss atomic capture
  {c = a; (a) += b;}
#pragma oss atomic capture
  {a %= b; c = a;}
#pragma oss atomic capture
  {c = a; a *= b;}
#pragma oss atomic capture
  {a -= b;c = a;}
#pragma oss atomic capture
  {c = a; a /= b;}
#pragma oss atomic capture
  {a &= b; c = a;}
#pragma oss atomic capture
  {c = a; a ^= b;}
#pragma oss atomic capture
  {a |= b; c = a;}
#pragma oss atomic capture
  {c = a; a <<= b;}
#pragma oss atomic capture
  {a >>= b; c = a;}
#pragma oss atomic capture
  {c = a; a = b + a;}
#pragma oss atomic capture
  {a = a * b; c = a;}
#pragma oss atomic capture
  {c = a; a = b - a;}
#pragma oss atomic capture
  {a = a / b; c = a;}
#pragma oss atomic capture
  {c = a; a = b & a;}
#pragma oss atomic capture
  {a = a ^ b; c = a;}
#pragma oss atomic capture
  {c = a; a = b | a;}
#pragma oss atomic capture
  {a = a << b; c = a;}
#pragma oss atomic capture
  {c = a; a = b >> a;}
#pragma oss atomic capture
  {c = a; a = foo();}
  // expected-error@+1 {{directive '#pragma oss atomic' cannot contain more than one 'capture' clause}}
#pragma oss atomic capture capture
  b = a /= b;

  return 0;
}

