#include "kmp.h"
#include <assert.h>
#include <limits>

struct spawn_data {
  void (*function)(void *);
  void *args;
  void (*completion_callback)(void *);
  void *completion_args;
};

static void nosv_spawn_run(nosv_task_t task) {
  spawn_data *metadata = (spawn_data*)nosv_get_task_metadata(task);
  metadata->function(metadata->args);
}

static void nosv_spawn_complete(nosv_task_t task) {
  spawn_data *metadata = (spawn_data*)nosv_get_task_metadata(task);
  if (metadata->completion_callback)
    metadata->completion_callback(metadata->completion_args);

  nosv_destroy(task, NOSV_DESTROY_NONE);
}

void nanos6_spawn_function(
	void (*function)(void *),
	void *args,
	void (*completion_callback)(void *),
	void *completion_args,
	char const *label
) {
  int res;
  nosv_task_type_t nosv_task_type;
  nosv_task_t nosv_task;

  res = nosv_type_init(
    &nosv_task_type,
    &nosv_spawn_run,
    NULL,
    &nosv_spawn_complete,
    label, NULL, NULL, NOSV_TYPE_INIT_NONE);
  KMP_ASSERT(res == 0);

  res = nosv_create(&nosv_task, nosv_task_type, sizeof(spawn_data), NOSV_CREATE_NONE);
  KMP_ASSERT(res == 0);

  spawn_data *metadata = (spawn_data*)nosv_get_task_metadata(nosv_task);
  metadata->function = function;
  metadata->args = args;
  metadata->completion_callback = completion_callback;
  metadata->completion_args = completion_args;

  res = nosv_submit(nosv_task, NOSV_SUBMIT_NONE);
  KMP_ASSERT(res == 0);
}

uint64_t nanos6_wait_for(uint64_t timeUs)
{
	if (timeUs == 0) {
		return 0;
	}

	uint64_t actualWaitTime;
	nosv_waitfor(timeUs * 1000, &(actualWaitTime));

	return actualWaitTime / (uint64_t) 1000;
}

void *nanos6_get_current_event_counter() {
  KMP_ASSERT(nosv_self() != NULL);
  return nosv_self();
}

void nanos6_increase_current_task_event_counter(void *event_counter,
                                                unsigned int increment) {
    // Atomic ops require signed, check if value is representable.
    KMP_DEBUG_ASSERT(sizeof(unsigned int) == sizeof(kmp_uint32));
    KMP_DEBUG_ASSERT(increment < std::numeric_limits<kmp_int32>::max());

    KMP_ASSERT(event_counter == nosv_self());

    nosv_increase_event_counter(increment);
}

void nanos6_decrease_task_event_counter(void *event_counter,
                                        unsigned int decrement) {
    // Atomic ops require signed, check if value is representable.
    KMP_DEBUG_ASSERT(sizeof(unsigned int) == sizeof(kmp_uint32));
    KMP_DEBUG_ASSERT(decrement < std::numeric_limits<kmp_int32>::max());

    nosv_decrease_event_counter((nosv_task_t)event_counter, decrement);
}

void nanos6_register_polling_service(char const *service_name,
                                     nanos6_polling_service_t service_function,
                                     void *service_data)
{
  abort();
}

void nanos6_unregister_polling_service(char const *service_name,
                                       nanos6_polling_service_t service_function,
                                       void *service_data)
{
  abort();
}

// This is supposed to be called in a MPI_Init
void nanos6_notify_task_event_counter_api() {
    int gtid = __kmp_entry_gtid();
    __kmp_enable_tasking_in_serial_mode(
      NULL, gtid,
      /*proxy=*/false, /*detachable=*/true, /*hidden_helper=*/false);
    KA_TRACE(5, ("event tasking  enabled\n"));
}
