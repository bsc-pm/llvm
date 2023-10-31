#ifndef INSTRUM_H_
#define INSTRUM_H_

#include <unistd.h>
#include "kmp_wrapper_getpid.h"

#if ENABLE_INSTRUMENTATION

#include <ovni.h>

extern int nosv_enable_ovni;

static inline void intrum_check_ovni() {
  if (nosv_enable_ovni)
    ovni_version_check();
}

#define INSTR_3ARG(name, mcv, ta, a, tb, b, tc, c)                \
	static inline void name(ta a, tb b, tc c)                 \
	{                                                         \
		if (!nosv_enable_ovni)                            \
			return;                                   \
		struct ovni_ev ev = {};                           \
		ovni_ev_set_clock(&ev, ovni_clock_now());         \
		ovni_ev_set_mcv(&ev, mcv);                        \
		ovni_payload_add(&ev, (uint8_t *) &a, sizeof(a)); \
		ovni_payload_add(&ev, (uint8_t *) &b, sizeof(b)); \
		ovni_payload_add(&ev, (uint8_t *) &c, sizeof(c)); \
		ovni_ev_emit(&ev);                                \
	}

#define INSTR_1ARG(name, mcv, ta, a)                              \
        static inline void name(ta a)                             \
        {                                                         \
		if (!nosv_enable_ovni)                            \
			return;                                   \
                struct ovni_ev ev = {};                           \
                ovni_ev_set_clock(&ev, ovni_clock_now());         \
                ovni_ev_set_mcv(&ev, mcv);                        \
                ovni_payload_add(&ev, (uint8_t *) &a, sizeof(a)); \
                ovni_ev_emit(&ev);                                \
        }

#define INSTR_2ARG(name, mcv, ta, a, tb, b)                       \
        static inline void name(ta a, tb b)                       \
        {                                                         \
		if (!nosv_enable_ovni)                            \
			return;                                   \
                struct ovni_ev ev = {};                           \
                ovni_ev_set_clock(&ev, ovni_clock_now());         \
                ovni_ev_set_mcv(&ev, mcv);                        \
                ovni_payload_add(&ev, (uint8_t *) &a, sizeof(a)); \
                ovni_payload_add(&ev, (uint8_t *) &b, sizeof(b)); \
                ovni_ev_emit(&ev);                                \
        }

#define INSTR_0ARG(name, mcv)                             \
        static inline void name(void)                     \
        {                                                 \
		if (!nosv_enable_ovni)                    \
			return;                           \
                struct ovni_ev ev = {};                   \
                ovni_ev_set_clock(&ev, ovni_clock_now()); \
                ovni_ev_set_mcv(&ev, mcv);                \
                ovni_ev_emit(&ev);                        \
        }

static inline void instr_thread_init()
{
	if (!nosv_enable_ovni)
		return;

	ovni_thread_init(__kmp_gettid());
}

static inline void instr_thread_end(void)
{
	struct ovni_ev ev = {};

	if (!nosv_enable_ovni)
		return;

	ovni_ev_set_clock(&ev, ovni_clock_now());
	ovni_ev_set_mcv(&ev, "OHe");
	ovni_ev_emit(&ev);

	// Flush the events to disk before killing the thread
	ovni_flush();
}

// A jumbo event is needed to encode a large label
static inline void instr_type_create(uint32_t id, const char *label)
{
        size_t bufsize, label_len, size_left;
        uint8_t buf[1024], *p;

	if (!nosv_enable_ovni)
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

                // FIXME: Print detailed truncation message
                fprintf(stderr, "The task label is too large, truncated\n");
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

#else // ENABLE_INSTRUMENTATION

static inline void intrum_check_ovni() {
  if (nosv_enable_ovni)
    fprintf(stderr, "WARNING: attempting to enable ovni in a runtime built without intrumentation\n");
}

#define INSTR_3ARG(name, mcv, ta, a, tb, b, tc, c)     \
	static inline void name(ta a, tb b, tc c)      \
	{                                              \
	}

#define INSTR_1ARG(name, mcv, ta, a)                   \
        static inline void name(ta a)                  \
        {                                              \
        }

#define INSTR_2ARG(name, mcv, ta, a, tb, b)            \
        static inline void name(ta a, tb b)            \
        {                                              \
        }

#define INSTR_0ARG(name, mcv)                             \
        static inline void name(void)                     \
        {                                                 \
        }

static inline void instr_thread_init() {}
static inline void instr_thread_end(void) {}
static inline void instr_type_create(uint32_t id, const char *label)
{
}

#endif // ENABLE_INSTRUMENTATION

INSTR_3ARG(instr_thread_execute, "OHx", int32_t, cpu, int32_t, creator_tid, uint64_t, tag)
INSTR_0ARG(instr_attached_enter, "PA[")
INSTR_0ARG(instr_attached_exit, "PA]")

INSTR_0ARG(instr_join_barrier_enter, "PBj")
INSTR_0ARG(instr_join_barrier_exit, "PBJ")
INSTR_0ARG(instr_barrier_enter, "PBb")
INSTR_0ARG(instr_barrier_exit, "PBB")
INSTR_0ARG(instr_tasking_barrier_enter, "PBt")
INSTR_0ARG(instr_tasking_barrier_exit, "PBT")
INSTR_0ARG(instr_spin_wait_enter, "PBs")
INSTR_0ARG(instr_spin_wait_exit, "PBS")

INSTR_0ARG(instr_for_static_enter, "PWs")
INSTR_0ARG(instr_for_static_exit, "PWS")
INSTR_0ARG(instr_for_dynamic_init_enter, "PWd")
INSTR_0ARG(instr_for_dynamic_init_exit, "PWD")
INSTR_0ARG(instr_for_dynamic_chunk_enter, "PWc")
INSTR_0ARG(instr_for_dynamic_chunk_exit, "PWC")
INSTR_0ARG(instr_single_enter, "PWi")
INSTR_0ARG(instr_single_exit, "PWI")

INSTR_0ARG(instr_release_deps_enter, "PTr")
INSTR_0ARG(instr_release_deps_exit, "PTR")
INSTR_0ARG(instr_taskwait_deps_enter, "PTw")
INSTR_0ARG(instr_taskwait_deps_exit, "PTW")
INSTR_0ARG(instr_invoke_task_enter, "PT[")
INSTR_0ARG(instr_invoke_task_exit, "PT]")
INSTR_0ARG(instr_invoke_task_if0_enter, "PTi")
INSTR_0ARG(instr_invoke_task_if0_exit, "PTI")
INSTR_0ARG(instr_task_alloc_enter, "PTa")
INSTR_0ARG(instr_task_alloc_exit, "PTA")
INSTR_0ARG(instr_task_schedule_enter, "PTs")
INSTR_0ARG(instr_task_schedule_exit, "PTS")
INSTR_0ARG(instr_taskwait_enter, "PTt")
INSTR_0ARG(instr_taskwait_exit, "PTT")
INSTR_0ARG(instr_taskyield_enter, "PTy")
INSTR_0ARG(instr_taskyield_exit, "PTY")
INSTR_0ARG(instr_task_dup_alloc_enter, "PTd")
INSTR_0ARG(instr_task_dup_alloc_exit, "PTD")
INSTR_0ARG(instr_check_deps_enter, "PTc")
INSTR_0ARG(instr_check_deps_exit, "PTC")
INSTR_0ARG(instr_taskgroup_enter, "PTg")
INSTR_0ARG(instr_taskgroup_exit, "PTG")


INSTR_2ARG(instr_task_create, "PPc", uint32_t, task_id, uint32_t, type_id)
INSTR_1ARG(instr_task_execute, "PPx", uint32_t, task_id)
INSTR_1ARG(instr_task_end, "PPe", uint32_t, task_id)

INSTR_1ARG(instr_ws_execute, "PQx", uint32_t, type_id)
INSTR_1ARG(instr_ws_end, "PQe", uint32_t, type_id)

#endif // INSTRUM_H_
