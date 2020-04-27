// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - -std=c++11 %s
// expected-no-diagnostics

// The implementantion of tentative parsing using ParseExpression or similar
// is not safe because Parser can perform token replacing

int main() {
  int x;
  auto l = [x](int value) {
    value = 4;
  };
}

