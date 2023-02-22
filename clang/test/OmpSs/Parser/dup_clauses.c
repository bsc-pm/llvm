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
#pragma oss task device(smp) device(cuda) // expected-error {{directive '#pragma oss task' cannot contain more than one 'device' clause}}
void task_label();
#pragma oss task ndrange(1,1,1) ndrange(1,1,1) // expected-error {{directive '#pragma oss task' cannot contain more than one 'ndrange' clause}}
void task_label();
#pragma oss task num_instances(1) num_instances(1) // expected-error {{directive '#pragma oss task' cannot contain more than one 'num_instances' clause}}
void task_label0();
#pragma oss task onto(1) onto(1) // expected-error {{directive '#pragma oss task' cannot contain more than one 'onto' clause}}
void task_label1();
#pragma oss task period(1) period(1) // expected-error {{directive '#pragma oss task' cannot contain more than one 'period' clause}}
void task_label2();
#pragma oss task affinity(1) affinity(1) // expected-error {{directive '#pragma oss task' cannot contain more than one 'affinity' clause}}
void task_label3();
#pragma oss task copy_deps copy_deps // expected-error {{directive '#pragma oss task' cannot contain more than one 'copy_deps' clause}}
void task_label7();

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
  // TODO: collapse is specia because it's preparsed
  // #pragma oss taskloop for collapse(2) grainsize(2)
  // for (int i = 0; i < 10; ++i)
  //     for (int j = 0; j < 10; ++j) {}
}
