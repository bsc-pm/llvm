// RUN: %if omp %{ %libomp-compile-and-run | FileCheck %s %}
// RUN: %if omp %{ %libomp-compile-and-run | %sort-threads \
// RUN:                         | FileCheck --check-prefix=THREADS %s %}
// RUN: %if ompv || ompv-free-agents %{ %libomp-compile-and-run | FileCheck %s --check-prefix=CHECK-OMPV %}
// RUN: %if ompv || ompv-free-agents %{ %libomp-compile-and-run | %sort-threads \
// RUN:                         | FileCheck --check-prefix=THREADS-OMPV %s %}
// REQUIRES: ompt
#include "callback.h"

int main() {
#pragma omp parallel num_threads(4)
  {
    print_ids(0);
    print_ids(1);
  }
  print_fuzzy_address(1);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: parallel_data initially not null
  // CHECK-NOT: 0: task_data initially not null
  // CHECK-NOT: 0: thread_data initially not null

  // Only check callback names, arguments are verified in THREADS below.
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin

  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin
  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end

  // Note that we cannot ensure that the worker threads have already called
  // barrier_end and implicit_task_end before parallel_end!

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end

  // THREADS: 0: NULL_POINTER=[[NULL:.*$]]
  // THREADS: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // THREADS-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]]
  // THREADS-SAME: parent_task_frame.exit=[[NULL]]
  // THREADS-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // THREADS-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4
  // THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}

  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 0
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 1
  // THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]]
  // THREADS-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: ompt_event_implicit_task_end
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // THREADS: {{^}}[[MASTER_ID]]: task level 0
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-SAME: exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1
  // THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // THREADS-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1
  // THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // THREADS-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1
  // THREADS-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // THREADS-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-NOT: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // THREADS-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // Check if libomp supports the callbacks for this test.
  // CHECK-OMPV-NOT: {{^}}0: Could not register callback

  // CHECK-OMPV: 0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-OMPV-NOT: 0: parallel_data initially not null
  // CHECK-OMPV-NOT: 0: task_data initially not null
  // CHECK-OMPV-NOT: 0: thread_data initially not null

  // Only check callback names, arguments are verified in THREADS-OMPV below.
  // CHECK-OMPV: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin

  // CHECK-OMPV-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin
  // CHECK-OMPV-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end

  // Note that we cannot ensure that the worker threads have already called
  // barrier_end and implicit_task_end before parallel_end!

  // CHECK-OMPV-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-OMPV-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin

  // CHECK-OMPV-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-OMPV-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin

  // CHECK-OMPV-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-OMPV-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin

  // CHECK-OMPV: {{^}}[[MASTER_ID]]: ompt_event_parallel_end

  // THREADS-OMPV: 0: NULL_POINTER=[[NULL:.*$]]
  // THREADS-OMPV: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-OMPV-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID]]
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // THREADS-OMPV-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]]
  // THREADS-OMPV-SAME: parent_task_frame.exit=[[NULL]]
  // THREADS-OMPV-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4
  // THREADS-OMPV-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}

  // THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 1
  // THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]]
  // THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-OMPV-NOT: ompt_event_implicit_task_end
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV-SAME: codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: task level 0
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV-SAME: exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV-SAME: codeptr_ra=(nil)
  // THREADS-OMPV: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS-OMPV: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-OMPV-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1
  // THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-OMPV-NOT: ompt_event_implicit_task_end
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS-OMPV: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-OMPV-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1
  // THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-OMPV-NOT: ompt_event_implicit_task_end
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS-OMPV: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin
  // THREADS-OMPV-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 0
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: task level 1
  // THREADS-OMPV-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // THREADS-OMPV-SAME: task_id=[[PARENT_TASK_ID]]
  // THREADS-OMPV-NOT: ompt_event_implicit_task_end
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // THREADS-OMPV-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS-OMPV: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS-OMPV-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  return 0;
}
