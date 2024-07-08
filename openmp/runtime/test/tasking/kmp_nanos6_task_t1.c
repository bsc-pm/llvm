// RUN: %libomp-compile && env OMP_NUM_THREADS='3' %libomp-run
// The runtime currently does not get dependency information from GCC.
// UNSUPPORTED: gcc

#if defined(_OPENMPV)

#include <stdio.h>
#include <omp.h>
#include "omp_my_sleep.h"

// detached untied
#define PTASK_FLAG_DETACHABLE 0x40

// OpenMP RTL interfaces
typedef unsigned long long kmp_uint64;
typedef long long kmp_int64;

typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;

// Compiler-generated code (emulation)
typedef struct ident {
  void* dummy; // not used in the library
} ident_t;

typedef enum kmp_event_type_t {
  KMP_EVENT_UNINITIALIZED = 0,
  KMP_EVENT_ALLOW_COMPLETION = 1
} kmp_event_type_t;

typedef struct {
  kmp_event_type_t type;
  union {
    void *task;
  } ed;
} kmp_event_t;

typedef struct shar { // shareds used in the task
} *pshareds;

typedef struct task {
  pshareds shareds;
  int(*routine)(int,struct task*);
  int part_id;
// void *destructor_thunk; // optional, needs flag setting if provided
// int priority; // optional, needs flag setting if provided
// ------------------------------
// privates used in the task:
  omp_event_handle_t evt;
} *ptask, kmp_task_t;

typedef struct DEP {
  size_t addr;
  size_t len;
  int flags;
} dep;

typedef int(* task_entry_t)( int, ptask );
typedef void *omp_task_type_t;

#ifdef __cplusplus
extern "C" {
#endif
extern int  __kmpc_global_thread_num(void *id_ref);
extern void __nosvc_register_task_info(omp_task_type_t *omp_task_type, void *label);
extern int** __nosvc_omp_task_alloc(id *loc, int gtid, int flags,
                                   size_t sz, size_t shar, task_entry_t rtn, omp_task_type_t*);
extern int __kmpc_omp_task_with_deps(id *loc, int gtid, ptask task, int nd,
               dep *dep_lst, int nd_noalias, dep *noalias_dep_lst);
extern int __kmpc_omp_task(id *loc, int gtid, kmp_task_t *task);
extern omp_event_handle_t __kmpc_task_allow_completion_event(
                              ident_t *loc_ref, int gtid, kmp_task_t *task);
int alpi_task_self(void **task);
int alpi_task_events_increase(void *task, uint64_t increment);
int alpi_task_events_decrease(void *task, uint64_t increment);
#ifdef __cplusplus
}
#endif

int checker1;
int volatile checker;
void *nanos6_event_counter = NULL;

// User's code, outlined into task entry
int task_entry(int gtid, ptask task) {
  alpi_task_self(&nanos6_event_counter);
  alpi_task_events_increase(nanos6_event_counter, 1);

  checker = 1;
  return 0;
}

int main() {
  int i, j, gtid = __kmpc_global_thread_num(NULL);
  int nt = omp_get_max_threads();
  ptask task;
  pshareds psh;
  checker = 0;
  checker1 = 0;
  omp_set_dynamic(0);
  #pragma omp parallel //num_threads(N)
  {
    #pragma omp master
    {
      // FIXME: Trick to make CodeGen to build first __nosvc_register_task_info
      // so extern will resolve to it instead of building a function declaration
      #pragma omp task
      {}
      #pragma omp taskwait
      int gtid = __kmpc_global_thread_num(NULL);
/*
      #pragma omp task depend(inout : nt)
      {}
*/
      omp_task_type_t omp_task_type;
      __nosvc_register_task_info(&omp_task_type, NULL);
      task = (ptask)__nosvc_omp_task_alloc(NULL,gtid,0,
                        sizeof(struct task),sizeof(struct shar),&task_entry, &omp_task_type);
      psh = task->shareds;

      dep sdep;
      sdep.addr = (size_t)&nt;
      sdep.len = 0L;
      sdep.flags = 3;

      __kmpc_omp_task_with_deps(NULL,gtid,task,1,&sdep,0,0);
      //__kmpc_omp_task(NULL, gtid, task);

      #pragma omp task depend(inout:nt)
      {
        checker = 2;
      }
      my_sleep(2.0);
      if (checker == 1) {
        ++checker1;
      }
      alpi_task_events_decrease(nanos6_event_counter, 1);
      #pragma omp taskwait
      if (checker == 2) {
        ++checker1;
      }
    } // end master
  } // end parallel
  // check results
  if (checker1 == 2) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed\n");
    return 1;
  }
}

#else

int main() {}

#endif
