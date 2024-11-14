#ifndef INSTRUM_H_
#define INSTRUM_H_

#include <unistd.h>
#include "kmp.h"
#include "kmp_wrapper_getpid.h"

enum instr_levels {
  INSTR_0 = 0, /* Disabled */
  INSTR_1 = 1, /* Enabled */
};

extern int ompv_instr_level;

#if ENABLE_INSTRUMENTATION

#include <ovni.h>

static inline void instr_check_ovni() {
  if (ompv_instr_level > 0)
    ovni_version_check();
}

#define INSTR_0ARG(l, name, mcv)                      \
  static inline void name(void)                       \
  {                                                   \
    if (ompv_instr_level < l)                         \
      return;                                         \
    struct ovni_ev ev = {};                           \
    ovni_ev_set_clock(&ev, ovni_clock_now());         \
    ovni_ev_set_mcv(&ev, mcv);                        \
    ovni_ev_emit(&ev);                                \
  }

#define INSTR_1ARG(l, name, mcv, ta, a)               \
  static inline void name(ta a)                       \
  {                                                   \
    if (ompv_instr_level < l)                         \
      return;                                         \
    struct ovni_ev ev = {};                           \
    ovni_ev_set_clock(&ev, ovni_clock_now());         \
    ovni_ev_set_mcv(&ev, mcv);                        \
    ovni_payload_add(&ev, (uint8_t *) &a, sizeof(a)); \
    ovni_ev_emit(&ev);                                \
  }

#define INSTR_2ARG(l, name, mcv, ta, a, tb, b)        \
  static inline void name(ta a, tb b)                 \
  {                                                   \
    if (ompv_instr_level < l)                         \
      return;                                         \
    struct ovni_ev ev = {};                           \
    ovni_ev_set_clock(&ev, ovni_clock_now());         \
    ovni_ev_set_mcv(&ev, mcv);                        \
    ovni_payload_add(&ev, (uint8_t *) &a, sizeof(a)); \
    ovni_payload_add(&ev, (uint8_t *) &b, sizeof(b)); \
    ovni_ev_emit(&ev);                                \
  }

#define INSTR_3ARG(l, name, mcv, ta, a, tb, b, tc, c) \
  static inline void name(ta a, tb b, tc c)           \
  {                                                   \
    if (ompv_instr_level < l)                         \
      return;                                         \
    struct ovni_ev ev = {};                           \
    ovni_ev_set_clock(&ev, ovni_clock_now());         \
    ovni_ev_set_mcv(&ev, mcv);                        \
    ovni_payload_add(&ev, (uint8_t *) &a, sizeof(a)); \
    ovni_payload_add(&ev, (uint8_t *) &b, sizeof(b)); \
    ovni_payload_add(&ev, (uint8_t *) &c, sizeof(c)); \
    ovni_ev_emit(&ev);                                \
  }

static inline void instr_thread_init()
{
  if (ompv_instr_level < 1)
    return;

  ovni_thread_init(__kmp_gettid());
  ovni_thread_require("openmp", "1.2.0");
}

static inline void instr_thread_end(void)
{
  struct ovni_ev ev = {};

  if (ompv_instr_level < 1)
    return;

  ovni_ev_set_clock(&ev, ovni_clock_now());
  ovni_ev_set_mcv(&ev, "OHe");
  ovni_ev_emit(&ev);

  // Flush the events to disk before killing the thread
  ovni_flush();
  ovni_thread_free();
}

// A jumbo event is needed to encode a large label
static inline void instr_type_create(uint32_t id, const char *label)
{
  size_t bufsize, label_len, size_left;
  uint8_t buf[1024], *p;

  if (ompv_instr_level < 1)
    return;

  p = buf;
  size_left = sizeof(buf);
  bufsize = 0;

  memcpy(p, &id, sizeof(id));
  p += sizeof(id);
  size_left -= sizeof(id);
  bufsize += sizeof(id);

  if (label == NULL)
    label = "";

  label_len = strlen(label);

  // Truncate the label if required
  if (label_len > size_left - 1) {
    // Maximum length of the label without the '\0'
    label_len = size_left - 1;
    fprintf(stderr, "The task label '%s' is too large, truncated\n", label);
  }

  memcpy(p, label, label_len);
  p += label_len;
  bufsize += label_len;

  // Always terminate the label
  *p = '\0';
  bufsize += 1;

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());
  ovni_ev_set_mcv(&ev, "POc");
  ovni_ev_jumbo_emit(&ev, buf, bufsize);
}

static inline void instr_attached_enter(void)
{
  if (ompv_instr_level < 1)
    return;

  ovni_thread_require("openmp", "1.1.0");

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());
  ovni_ev_set_mcv(&ev, "PA[");
  ovni_ev_emit(&ev);
}

static inline void instr_for_static_enter(kmp_int32 flags)
{
  if (ompv_instr_level < 1)
    return;

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());

  if (flags & KMP_IDENT_WORK_SECTIONS)
    ovni_ev_set_mcv(&ev, "PWe");
  else if (flags & KMP_IDENT_WORK_DISTRIBUTE)
    ovni_ev_set_mcv(&ev, "PWd");
  else /* static for */
    ovni_ev_set_mcv(&ev, "PWs");

  ovni_ev_emit(&ev);
}

static inline void instr_for_static_exit(kmp_int32 flags)
{
  if (ompv_instr_level < 1)
    return;

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());

  if (flags & KMP_IDENT_WORK_SECTIONS)
    ovni_ev_set_mcv(&ev, "PWE");
  else if (flags & KMP_IDENT_WORK_DISTRIBUTE)
    ovni_ev_set_mcv(&ev, "PWD");
  else /* static for */
    ovni_ev_set_mcv(&ev, "PWS");

  ovni_ev_emit(&ev);
}

static inline void instr_microtask_enter(microtask_t t)
{
  if (ompv_instr_level < 1)
    return;

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());

  if (t == (microtask_t)__kmp_teams_master) {
    /* Internal microtask */
    ovni_ev_set_mcv(&ev, "PMi");
  } else {
    /* "User" code */
    ovni_ev_set_mcv(&ev, "PMu");
  }

  ovni_ev_emit(&ev);
}

static inline void instr_microtask_exit(microtask_t t)
{
  if (ompv_instr_level < 1)
    return;

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());

  if (t == (microtask_t)__kmp_teams_master) {
    /* Internal microtask */
    ovni_ev_set_mcv(&ev, "PMI");
  } else {
    /* "User" code */
    ovni_ev_set_mcv(&ev, "PMU");
  }

  ovni_ev_emit(&ev);
}

#else // ENABLE_INSTRUMENTATION

static inline void instr_check_ovni() {
  if (ompv_instr_level != 0) {
    fprintf(stderr, "ERROR: attempting to enable ovni in a runtime built without intrumentation\n");
    abort();
  }
}

#define INSTR_0ARG(l, name, mcv)                      \
  static inline void name(void) {}

#define INSTR_1ARG(l, name, mcv, ta, a)               \
  static inline void name(ta a) {}

#define INSTR_2ARG(l, name, mcv, ta, a, tb, b)        \
  static inline void name(ta a, tb b) {}

#define INSTR_3ARG(l, name, mcv, ta, a, tb, b, tc, c) \
  static inline void name(ta a, tb b, tc c) {}


static inline void instr_thread_init() {}
static inline void instr_thread_end(void) {}
static inline void instr_type_create(uint32_t id, const char *label) {}
static inline void instr_attached_enter(void) {}
static inline void instr_for_static_enter(kmp_int32 flags) {}
static inline void instr_for_static_exit(kmp_int32 flags) {}
static inline void instr_microtask_enter(microtask_t t) {}
static inline void instr_microtask_exit(microtask_t t) {}

#endif // ENABLE_INSTRUMENTATION

INSTR_3ARG(1, instr_thread_execute, "OHx", int32_t, cpu, int32_t, creator_tid, uint64_t, tag)
INSTR_0ARG(1, instr_attached_exit, "PA]")

// "B" for barrier
INSTR_0ARG(1, instr_barrier_enter, "PBb") // plain
INSTR_0ARG(1, instr_barrier_exit, "PBB")
INSTR_0ARG(1, instr_join_barrier_enter, "PBj")
INSTR_0ARG(1, instr_join_barrier_exit, "PBJ")
INSTR_0ARG(1, instr_fork_barrier_enter, "PBf")
INSTR_0ARG(1, instr_fork_barrier_exit, "PBF")
INSTR_0ARG(1, instr_tasking_barrier_enter, "PBt")
INSTR_0ARG(1, instr_tasking_barrier_exit, "PBT")
INSTR_0ARG(1, instr_spin_wait_enter, "PBs") // disabled in emulation
INSTR_0ARG(1, instr_spin_wait_exit, "PBS")

// "W" for work-distribution
INSTR_0ARG(1, instr_single_enter, "PWi")
INSTR_0ARG(1, instr_single_exit, "PWI")
INSTR_0ARG(1, instr_for_dynamic_init_enter, "PWy")
INSTR_0ARG(1, instr_for_dynamic_init_exit, "PWY")
INSTR_0ARG(1, instr_for_dynamic_chunk_enter, "PWc")
INSTR_0ARG(1, instr_for_dynamic_chunk_exit, "PWC")
/* instr_for_static_enter may emit PWe PWd and PWs depending on the
 * work-distribute type: section, distribute or for loop. Then
 * instr_for_static_exit will emit the correspoding PWE PWD and PWS. */

// "C" for kmp_csupport.cpp, the C API
INSTR_0ARG(1, instr_fork_enter, "PCf")
INSTR_0ARG(1, instr_fork_exit, "PCF")
INSTR_0ARG(1, instr_init_enter, "PCi")
INSTR_0ARG(1, instr_init_exit, "PCI")

// "I" for critical
INSTR_0ARG(1, instr_critical_acquire_enter, "PIa")
INSTR_0ARG(1, instr_critical_acquire_exit,  "PIA")
INSTR_0ARG(1, instr_critical_release_enter, "PIr")
INSTR_0ARG(1, instr_critical_release_exit,  "PIR")
INSTR_0ARG(1, instr_critical_region_enter,  "PI[")
INSTR_0ARG(1, instr_critical_region_exit,   "PI]")

// "M" for microtasks
/* instr_microtask_enter and instr_microtask_exit */

// "H" for threads
INSTR_0ARG(1, instr_launch_thread_enter, "PH[")
INSTR_0ARG(1, instr_launch_thread_exit, "PH]")

INSTR_0ARG(1, instr_release_deps_enter, "PTr")
INSTR_0ARG(1, instr_release_deps_exit, "PTR")
INSTR_0ARG(1, instr_taskwait_deps_enter, "PTw")
INSTR_0ARG(1, instr_taskwait_deps_exit, "PTW")
INSTR_0ARG(1, instr_invoke_task_enter, "PT[")
INSTR_0ARG(1, instr_invoke_task_exit, "PT]")
INSTR_0ARG(1, instr_invoke_task_if0_enter, "PTi")
INSTR_0ARG(1, instr_invoke_task_if0_exit, "PTI")
INSTR_0ARG(1, instr_task_alloc_enter, "PTa")
INSTR_0ARG(1, instr_task_alloc_exit, "PTA")
INSTR_0ARG(1, instr_task_schedule_enter, "PTs")
INSTR_0ARG(1, instr_task_schedule_exit, "PTS")
INSTR_0ARG(1, instr_taskwait_enter, "PTt")
INSTR_0ARG(1, instr_taskwait_exit, "PTT")
INSTR_0ARG(1, instr_taskyield_enter, "PTy")
INSTR_0ARG(1, instr_taskyield_exit, "PTY")
INSTR_0ARG(1, instr_task_dup_alloc_enter, "PTd")
INSTR_0ARG(1, instr_task_dup_alloc_exit, "PTD")
INSTR_0ARG(1, instr_check_deps_enter, "PTc")
INSTR_0ARG(1, instr_check_deps_exit, "PTC")
INSTR_0ARG(1, instr_taskgroup_enter, "PTg")
INSTR_0ARG(1, instr_taskgroup_exit, "PTG")

INSTR_2ARG(1, instr_task_create, "PPc", uint32_t, task_id, uint32_t, type_id)
INSTR_1ARG(1, instr_task_execute, "PPx", uint32_t, task_id)
INSTR_1ARG(1, instr_task_end, "PPe", uint32_t, task_id)

INSTR_1ARG(1, instr_ws_execute, "PQx", uint32_t, type_id)
INSTR_1ARG(1, instr_ws_end, "PQe", uint32_t, type_id)

#endif // INSTRUM_H_
