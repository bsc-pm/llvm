#ifndef INSTRUM_H_
#define INSTRUM_H_

#include <unistd.h>
#include "kmp_wrapper_getpid.h"

enum instr_levels {
  INSTR_NONE  = 0,
  INSTR_BASIC = 1,
  INSTR_SS    = 2, /* Subsystems */
  INSTR_WS    = 3, /* Worksharings */
};

#if ENABLE_INSTRUMENTATION

#include <ovni.h>

extern int ompv_instr_level;

static inline void intrum_check_ovni() {
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
  if (ompv_instr_level < INSTR_BASIC)
    return;

  ovni_thread_init(__kmp_gettid());
  ovni_thread_require("openmp", "1.0.0");
}

static inline void instr_thread_end(void)
{
  struct ovni_ev ev = {};

  if (ompv_instr_level < INSTR_BASIC)
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

  if (ompv_instr_level < INSTR_WS)
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
  if (ompv_instr_level < INSTR_SS)
    return;

  ovni_thread_require("openmp", "1.0.0");

  struct ovni_ev ev = {};
  ovni_ev_set_clock(&ev, ovni_clock_now());
  ovni_ev_set_mcv(&ev, "PA[");
  ovni_ev_emit(&ev);
}

#else // ENABLE_INSTRUMENTATION

static inline void intrum_check_ovni() {
  if (ompv_instr_level != INSTR_NONE) {
    fprintf(stderr, "WARNING: attempting to enable ovni in a runtime built without intrumentation\n");
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

#endif // ENABLE_INSTRUMENTATION

INSTR_3ARG(INSTR_BASIC, instr_thread_execute, "OHx", int32_t, cpu, int32_t, creator_tid, uint64_t, tag)
INSTR_0ARG(INSTR_SS, instr_attached_exit, "PA]")

INSTR_0ARG(INSTR_SS, instr_join_barrier_enter, "PBj")
INSTR_0ARG(INSTR_SS, instr_join_barrier_exit, "PBJ")
INSTR_0ARG(INSTR_SS, instr_barrier_enter, "PBb")
INSTR_0ARG(INSTR_SS, instr_barrier_exit, "PBB")
INSTR_0ARG(INSTR_SS, instr_tasking_barrier_enter, "PBt")
INSTR_0ARG(INSTR_SS, instr_tasking_barrier_exit, "PBT")
INSTR_0ARG(INSTR_SS, instr_spin_wait_enter, "PBs")
INSTR_0ARG(INSTR_SS, instr_spin_wait_exit, "PBS")

INSTR_0ARG(INSTR_SS, instr_for_static_enter, "PWs")
INSTR_0ARG(INSTR_SS, instr_for_static_exit, "PWS")
INSTR_0ARG(INSTR_SS, instr_for_dynamic_init_enter, "PWd")
INSTR_0ARG(INSTR_SS, instr_for_dynamic_init_exit, "PWD")
INSTR_0ARG(INSTR_SS, instr_for_dynamic_chunk_enter, "PWc")
INSTR_0ARG(INSTR_SS, instr_for_dynamic_chunk_exit, "PWC")
INSTR_0ARG(INSTR_SS, instr_single_enter, "PWi")
INSTR_0ARG(INSTR_SS, instr_single_exit, "PWI")

// "C" for kmp_csupport, the C API
INSTR_0ARG(INSTR_SS, instr_microtask_enter, "PCm")
INSTR_0ARG(INSTR_SS, instr_microtask_exit, "PCM")
INSTR_0ARG(INSTR_SS, instr_fork_enter, "PCf")
INSTR_0ARG(INSTR_SS, instr_fork_exit, "PCF")
INSTR_0ARG(INSTR_SS, instr_critical_enter, "PCc")
INSTR_0ARG(INSTR_SS, instr_critical_exit, "PCC")
INSTR_0ARG(INSTR_SS, instr_end_critical_enter, "PCe")
INSTR_0ARG(INSTR_SS, instr_end_critical_exit, "PCE")

INSTR_0ARG(INSTR_SS, instr_release_deps_enter, "PTr")
INSTR_0ARG(INSTR_SS, instr_release_deps_exit, "PTR")
INSTR_0ARG(INSTR_SS, instr_taskwait_deps_enter, "PTw")
INSTR_0ARG(INSTR_SS, instr_taskwait_deps_exit, "PTW")
INSTR_0ARG(INSTR_SS, instr_invoke_task_enter, "PT[")
INSTR_0ARG(INSTR_SS, instr_invoke_task_exit, "PT]")
INSTR_0ARG(INSTR_SS, instr_invoke_task_if0_enter, "PTi")
INSTR_0ARG(INSTR_SS, instr_invoke_task_if0_exit, "PTI")
INSTR_0ARG(INSTR_SS, instr_task_alloc_enter, "PTa")
INSTR_0ARG(INSTR_SS, instr_task_alloc_exit, "PTA")
INSTR_0ARG(INSTR_SS, instr_task_schedule_enter, "PTs")
INSTR_0ARG(INSTR_SS, instr_task_schedule_exit, "PTS")
INSTR_0ARG(INSTR_SS, instr_taskwait_enter, "PTt")
INSTR_0ARG(INSTR_SS, instr_taskwait_exit, "PTT")
INSTR_0ARG(INSTR_SS, instr_taskyield_enter, "PTy")
INSTR_0ARG(INSTR_SS, instr_taskyield_exit, "PTY")
INSTR_0ARG(INSTR_SS, instr_task_dup_alloc_enter, "PTd")
INSTR_0ARG(INSTR_SS, instr_task_dup_alloc_exit, "PTD")
INSTR_0ARG(INSTR_SS, instr_check_deps_enter, "PTc")
INSTR_0ARG(INSTR_SS, instr_check_deps_exit, "PTC")
INSTR_0ARG(INSTR_SS, instr_taskgroup_enter, "PTg")
INSTR_0ARG(INSTR_SS, instr_taskgroup_exit, "PTG")

INSTR_2ARG(INSTR_WS, instr_task_create, "PPc", uint32_t, task_id, uint32_t, type_id)
INSTR_1ARG(INSTR_WS, instr_task_execute, "PPx", uint32_t, task_id)
INSTR_1ARG(INSTR_WS, instr_task_end, "PPe", uint32_t, task_id)

INSTR_1ARG(INSTR_WS, instr_ws_execute, "PQx", uint32_t, type_id)
INSTR_1ARG(INSTR_WS, instr_ws_end, "PQe", uint32_t, type_id)

#endif // INSTRUM_H_
