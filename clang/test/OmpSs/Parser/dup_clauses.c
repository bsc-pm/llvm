// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s

#pragma oss task if(1) if(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'if' clause}}
void task_if();
#pragma oss task final(1) final(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'final' clause}}
void task_final();
#pragma oss task cost(1) cost(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'cost' clause}}
void task_cost();
#pragma oss task priority(1) priority(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'priority' clause}}
void task_priority();
#pragma oss task wait wait // expected-error {{directive '#pragma oss task' cannot contain more than one 'wait' clause}}
void task_wait();
#pragma oss task label("L1") label("L2") // expected-error {{directive '#pragma oss task' cannot contain more than one 'label' clause}}
void task_label();

void bar() {
  #pragma oss task if(1) if(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'if' clause}}
  #pragma oss task final(1) final(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'final' clause}}
  #pragma oss task cost(1) cost(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'cost' clause}}
  #pragma oss task priority(1) priority(0) // expected-error {{directive '#pragma oss task' cannot contain more than one 'priority' clause}}
  #pragma oss task wait wait // expected-error {{directive '#pragma oss task' cannot contain more than one 'wait' clause}}
  #pragma oss task default(none) default(shared) // expected-error {{directive '#pragma oss task' cannot contain more than one 'default' clause}}
  #pragma oss task label("L1") label("L2") // expected-error {{directive '#pragma oss task' cannot contain more than one 'label' clause}}
  {}
  #pragma oss task for chunksize(0) chunksize(0) // expected-error {{directive '#pragma oss task for' cannot contain more than one 'chunksize' clause}}
  for (int i = 0; i < 10; ++i) {}
  #pragma oss taskloop grainsize(0) grainsize(0) // expected-error {{directive '#pragma oss taskloop' cannot contain more than one 'grainsize' clause}}
  for (int i = 0; i < 10; ++i) {}
  #pragma oss taskloop for chunksize(0) chunksize(0) // expected-error {{directive '#pragma oss taskloop for' cannot contain more than one 'chunksize' clause}}
  for (int i = 0; i < 10; ++i) {}
  #pragma oss taskloop for grainsize(0) grainsize(0) // expected-error {{directive '#pragma oss taskloop for' cannot contain more than one 'grainsize' clause}}
  for (int i = 0; i < 10; ++i) {}
}
