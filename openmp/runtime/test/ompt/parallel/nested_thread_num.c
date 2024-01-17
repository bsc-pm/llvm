// RUN: %if omp %{ %libomp-compile-and-run | FileCheck %s %}
// RUN: %if omp %{ %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=THREADS %s %}
// RUN: %if ompv || ompv-free-agents %{ %libomp-compile-and-run | FileCheck %s --check-prefix=CHECK-OMPV %}
// RUN: %if ompv || ompv-free-agents %{ %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=THREADS-OMPV %s %}
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#define TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN
#include "callback.h"
#include <omp.h>
#include <unistd.h>

int main() {
  int condition = 0;
  omp_set_nested(1);
  print_frame(0);

#pragma omp parallel num_threads(2)
  {
    print_frame_from_outlined_fn(1);
    print_ids(0);
    print_ids(1);
    print_frame(0);

// get all implicit task events before starting nested:
#pragma omp barrier

#pragma omp parallel num_threads(2)
    {
      print_frame_from_outlined_fn(1);
      print_ids(0);
      print_ids(1);
      print_ids(2);
      print_frame(0);
      OMPT_SIGNAL(condition);
      OMPT_WAIT(condition, 4);
#pragma omp barrier
      print_fuzzy_address(1);
      print_ids(0);
    }
    print_fuzzy_address(2);
    print_ids(0);
  }
  print_fuzzy_address(3);

  return 0;
}
// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// make sure initial data pointers are null
// CHECK-NOT: 0: parallel_data initially not null
// CHECK-NOT: 0: task_data initially not null
// CHECK-NOT: 0: thread_data initially not null

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]],
// CHECK-SAME: parent_task_frame.exit=[[NULL]],
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]],
// CHECK-SAME: requested_team_size=2,
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// CHECK-SAME: invoker=[[PARALLEL_INVOKER:[0-9]+]]

// CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end:

// Note that we cannot ensure that the worker threads have already called
// barrier_end and implicit_task_end before parallel_end!

// CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:


// CHECK: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], 
// CHECK-SAME: task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
// CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

// THREADS: {{^}}0: NULL_POINTER=[[NULL:.*$]]
// THREADS: __builtin_frame_address(0)=[[MAIN_REENTER:0x[0-f]+]]
// THREADS: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
// THREADS-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], 
// THREADS-SAME: parent_task_frame.exit=[[NULL]],
// THREADS-SAME: parent_task_frame.reenter=0x{{[0-f]+}},
// THREADS-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=2,
// THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// THREADS-SAME: invoker=[[PARALLEL_INVOKER:[0-9]+]]

// nested parallel masters
// THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// THREADS-SAME: parallel_id=[[PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]],
// THREADS-SAME: team_size=2, thread_num=0

// THREADS: __builtin_frame_address({{.}})=[[EXIT:0x[0-f]+]]

// THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], 
// THREADS-SAME: reenter_frame=[[NULL]], 
// THREADS-SAME: thread_num=0

// THREADS: {{^}}[[MASTER_ID]]: task level 1:
// THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]], 
// THREADS-SAME: task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], 
// THREADS-SAME: reenter_frame=0x{{[0-f]+}}

// THREADS: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin:
// THREADS-SAME: parent_task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-SAME: parent_task_frame.exit=[[EXIT]],
// THREADS-SAME: parent_task_frame.reenter=0x{{[0-f]+}},
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], 
// THREADS-SAME: requested_team_size=2,
// THREADS-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// THREADS-SAME: invoker=[[PARALLEL_INVOKER]]

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]],
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]], team_size=2, 
// THREADS-SAME: thread_num=0

// THREADS: __builtin_frame_address({{.}})=[[NESTED_EXIT:0x[0-f]+]]

// THREADS: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-SAME:  parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: exit_frame=[[NESTED_EXIT]], reenter_frame=[[NULL]], 
// THREADS-SAME: thread_num=0

// THREADS: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[PARALLEL_ID]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]],
// THREADS-SAME: reenter_frame=0x{{[0-f]+}}

// THREADS: {{^}}[[MASTER_ID]]: task level 2:
// THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], 
// THREADS-SAME: reenter_frame=0x{{[0-f]+}}

// THREADS: __builtin_frame_address(0)=[[NESTED_REENTER:0x[0-f]+]]

// THREADS-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
// explicit barrier

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: codeptr_ra=[[BARRIER_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-SAME:  parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: exit_frame=[[NESTED_EXIT]], reenter_frame=0x{{[0-f]+}}

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[BARRIER_RETURN_ADDRESS]]

// THREADS: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: exit_frame=[[NESTED_EXIT]], reenter_frame=[[NULL]]

// implicit barrier
// THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: exit_frame=[[NULL]], reenter_frame=[[NULL]]

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]],
// THREADS-SAME: invoker=[[PARALLEL_INVOKER]],
// THREADS-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]

// THREADS-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end

// THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], 
// THREADS-SAME: reenter_frame=[[NULL]]

// implicit barrier
// THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]],
// THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NULL]], 
// THREADS-SAME: reenter_frame=[[NULL]]

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]],
// THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
// THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]],
// THREADS-SAME: invoker=[[PARALLEL_INVOKER]], 
// THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

// Worker of first nesting level

// THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// THREADS-SAME: parallel_id=[[PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]], team_size=2, 
// THREADS-SAME: thread_num=[[OUTER_THREADNUM:[0-9]+]]

// THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-SAME: thread_num=[[OUTER_THREADNUM]]

// THREADS: {{^}}[[THREAD_ID]]: task level 1:
// THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[PARENT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin:
// THREADS-SAME: parent_task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-SAME: parent_task_frame.exit={{0x[0-f]+}},
// THREADS-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=2,
// THREADS-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}},
// THREADS-SAME: invoker=[[PARALLEL_INVOKER]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]],
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]], team_size=2,
// THREADS-SAME: thread_num=[[INNER_THREADNUM:[0-9]+]]

// THREADS: {{^}}[[THREAD_ID]]: task level 0:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-SAME: thread_num=[[INNER_THREADNUM]]

// THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[PARALLEL_ID]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-SAME: thread_num=[[OUTER_THREADNUM]]

// THREADS: {{^}}[[THREAD_ID]]: task level 2:
// THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[PARENT_TASK_ID]]

// THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]

// THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// nested parallel worker threads

// THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
// THREADS-SAME: thread_num=[[THREADNUM:[0-9]+]]

// THREADS: {{^}}[[THREAD_ID]]: task level 0:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]]
// THREADS-SAME: thread_num=[[THREADNUM]]

// can't reliably tell which parallel region is the parent...

// THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}},
// THREADS-SAME: task_id={{[0-9]+}}
// THREADS-SAME: thread_num={{[01]}}

// THREADS: {{^}}[[THREAD_ID]]: task level 2:
// THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[PARENT_TASK_ID]]
// THREADS-SAME: thread_num=0

// THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// other nested parallel worker threads

// THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]],
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
// THREADS-SAME: thread_num=[[THREADNUM:[0-9]+]]

// THREADS: {{^}}[[THREAD_ID]]: task level 0:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]]
// THREADS-SAME: thread_num=[[THREADNUM]]

// can't reliably tell which parallel region is the parent...

// THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}},
// THREADS-SAME: task_id={{[0-9]+}}
// THREADS-SAME: thread_num={{[01]}}

// THREADS: {{^}}[[THREAD_ID]]: task level 2:
// THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[PARENT_TASK_ID]]
// THREADS-SAME: thread_num=0

// THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-SAME: task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// Check if libomp supports the callbacks for this test.
// CHECK-OMPV-NOT: {{^}}0: Could not register callback

// CHECK-OMPV: 0: NULL_POINTER=[[NULL:.*$]]

// make sure initial data pointers are null
// CHECK-OMPV-NOT: 0: parallel_data initially not null
// CHECK-OMPV-NOT: 0: task_data initially not null
// CHECK-OMPV-NOT: 0: thread_data initially not null

// CHECK-OMPV: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
// CHECK-OMPV-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]],
// CHECK-OMPV-SAME: parent_task_frame.exit=[[NULL]],
// CHECK-OMPV-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-OMPV-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]],
// CHECK-OMPV-SAME: requested_team_size=2,
// CHECK-OMPV-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// CHECK-OMPV-SAME: invoker=[[PARALLEL_INVOKER:[0-9]+]]

// CHECK-OMPV-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// CHECK-OMPV-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end:

// Note that we cannot ensure that the worker threads have already called
// barrier_end and implicit_task_end before parallel_end!

// CHECK-OMPV-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-OMPV-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:


// CHECK-OMPV: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], 
// CHECK-OMPV-SAME: task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
// CHECK-OMPV: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

// THREADS-OMPV: {{^}}0: NULL_POINTER=[[NULL:.*$]]
// THREADS-OMPV: __builtin_frame_address(0)=[[MAIN_REENTER:0x[0-f]+]]
// THREADS-OMPV: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
// THREADS-OMPV-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], 
// THREADS-OMPV-SAME: parent_task_frame.exit=[[NULL]],
// THREADS-OMPV-SAME: parent_task_frame.reenter=0x{{[0-f]+}},
// THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=2,
// THREADS-OMPV-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// THREADS-OMPV-SAME: invoker=[[PARALLEL_INVOKER:[0-9]+]]

// nested parallel masters
// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]],
// THREADS-OMPV-SAME: team_size=2, thread_num=0

// THREADS-OMPV: __builtin_frame_address({{.}})=[[EXIT:0x[0-f]+]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], 
// THREADS-OMPV-SAME: reenter_frame=[[NULL]], 
// THREADS-OMPV-SAME: thread_num=0

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 1:
// THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]], 
// THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], 
// THREADS-OMPV-SAME: reenter_frame=0x{{[0-f]+}}

// THREADS-OMPV: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin:
// THREADS-OMPV-SAME: parent_task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-OMPV-SAME: parent_task_frame.exit=[[EXIT]],
// THREADS-OMPV-SAME: parent_task_frame.reenter=0x{{[0-f]+}},
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], 
// THREADS-OMPV-SAME: requested_team_size=2,
// THREADS-OMPV-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// THREADS-OMPV-SAME: invoker=[[PARALLEL_INVOKER]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]], team_size=2, 
// THREADS-OMPV-SAME: thread_num=0

// THREADS-OMPV: __builtin_frame_address({{.}})=[[NESTED_EXIT:0x[0-f]+]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-OMPV-SAME:  parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: exit_frame=[[NESTED_EXIT]], reenter_frame=[[NULL]], 
// THREADS-OMPV-SAME: thread_num=0

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]],
// THREADS-OMPV-SAME: reenter_frame=0x{{[0-f]+}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 2:
// THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], 
// THREADS-OMPV-SAME: reenter_frame=0x{{[0-f]+}}

// THREADS-OMPV: __builtin_frame_address(0)=[[NESTED_REENTER:0x[0-f]+]]

// THREADS-OMPV-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
// explicit barrier

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: codeptr_ra=[[BARRIER_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-OMPV-SAME:  parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: exit_frame=[[NESTED_EXIT]], reenter_frame=0x{{[0-f]+}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[BARRIER_RETURN_ADDRESS]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: exit_frame=[[NESTED_EXIT]], reenter_frame=[[NULL]]

// implicit barrier
// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: exit_frame=[[NULL]], reenter_frame=[[NULL]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: codeptr_ra=(nil)

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: invoker=[[PARALLEL_INVOKER]],
// THREADS-OMPV-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]

// THREADS-OMPV-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], 
// THREADS-OMPV-SAME: reenter_frame=[[NULL]]

// implicit barrier
// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NULL]], 
// THREADS-OMPV-SAME: reenter_frame=[[NULL]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: codeptr_ra=(nil)

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
// THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]],
// THREADS-OMPV-SAME: invoker=[[PARALLEL_INVOKER]], 
// THREADS-OMPV-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

// THREADS-OMPV: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

// Worker of first nesting level

// THREADS-OMPV: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]], team_size=2, 
// THREADS-OMPV-SAME: thread_num=[[OUTER_THREADNUM:[0-9]+]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-OMPV-SAME: thread_num=[[OUTER_THREADNUM]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1:
// THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin:
// THREADS-OMPV-SAME: parent_task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-OMPV-SAME: parent_task_frame.exit={{0x[0-f]+}},
// THREADS-OMPV-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=2,
// THREADS-OMPV-SAME: codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}},
// THREADS-OMPV-SAME: invoker=[[PARALLEL_INVOKER]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]], team_size=2,
// THREADS-OMPV-SAME: thread_num=[[INNER_THREADNUM:[0-9]+]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]],
// THREADS-OMPV-SAME: thread_num=[[INNER_THREADNUM]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[PARALLEL_ID]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], 
// THREADS-OMPV-SAME: thread_num=[[OUTER_THREADNUM]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 2:
// THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]

// THREADS-OMPV-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_parallel_end:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]

// THREADS-OMPV-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// nested parallel worker threads

// THREADS-OMPV: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
// THREADS-OMPV-SAME: thread_num=[[THREADNUM:[0-9]+]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]]
// THREADS-OMPV-SAME: thread_num=[[THREADNUM]]

// can't reliably tell which parallel region is the parent...

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}},
// THREADS-OMPV-SAME: task_id={{[0-9]+}}
// THREADS-OMPV-SAME: thread_num={{[01]}}

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 2:
// THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]
// THREADS-OMPV-SAME: thread_num=0

// THREADS-OMPV-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// other nested parallel worker threads

// THREADS-OMPV: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]],
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
// THREADS-OMPV-SAME: thread_num=[[THREADNUM:[0-9]+]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]]
// THREADS-OMPV-SAME: thread_num=[[THREADNUM]]

// can't reliably tell which parallel region is the parent...

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}},
// THREADS-OMPV-SAME: task_id={{[0-9]+}}
// THREADS-OMPV-SAME: thread_num={{[01]}}

// THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 2:
// THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]
// THREADS-OMPV-SAME: thread_num=0

// THREADS-OMPV-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin:
// THREADS-OMPV-SAME: parallel_id=[[NESTED_PARALLEL_ID]], 
// THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

// THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end:
// THREADS-OMPV-SAME: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

