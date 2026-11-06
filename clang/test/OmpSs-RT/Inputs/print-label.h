// Print the label of the running task. NODES uses it as nOS-V task type label.
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void *nosv_self(void);
void *nosv_get_task_type(void *task);
const char *nosv_get_task_type_label(void *type);
#ifdef __cplusplus
}
#endif

static void print_task_label(void) {
  fprintf(stderr, "%s\n",
          nosv_get_task_type_label(nosv_get_task_type(nosv_self())));
}
