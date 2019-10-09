// RUN: %clang_cc1 -verify -x c++ -std=c++11 -fompss-2 -ferror-limit 100 -o - %s

void bar() {
  #pragma oss task
  {
    auto l = []() {
        return 1;
        break; // expected-error {{'break' statement not in loop or switch statement}}
        goto label; // expected-error {{use of undeclared label 'label'}}
    };
label:
    return; // expected-error {{invalid branch from OmpSs-2 structured block}}
    break; // expected-error {{'break' statement not in loop or switch statement}}
    goto label;
  }
  auto l = []() {
    #pragma oss task
    {
      for (int i = 0; i < 10; ++i)
      #pragma oss task
      {
        continue; // expected-error {{'continue' statement not in loop statement}}
        return 2; // expected-error {{invalid branch from OmpSs-2 structured block}}
      }
      return 3; // expected-error {{invalid branch from OmpSs-2 structured block}}
      break; // expected-error {{'break' statement not in loop or switch statement}}
      goto label; // expected-error {{use of undeclared label 'label'}}
    }
label:
    return 4;
  };
}
