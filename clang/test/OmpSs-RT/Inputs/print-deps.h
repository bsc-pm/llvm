// Print "start:<address>" for every 1-dimensional dependency registered by a
// task, then forward to the runtime. Works with any OmpSs-2 runtime.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PRINT_DEPINFO1(NAME)                                                   \
  void NAME(void *handler, int symbol_index, const char *region_text,          \
            void *base_address, long dim1size, long dim1start, long dim1end) { \
    typedef void (*fn_t)(void *, int, const char *, void *, long, long, long); \
    static fn_t real;                                                          \
    if (!real)                                                                 \
      real = (fn_t)dlsym(RTLD_NEXT, #NAME);                                    \
    fprintf(stderr, "start:%p\n", (char *)base_address + dim1start);           \
    real(handler, symbol_index, region_text, base_address, dim1size,           \
         dim1start, dim1end);                                                  \
  }

PRINT_DEPINFO1(nanos6_register_region_read_depinfo1)
PRINT_DEPINFO1(nanos6_register_region_write_depinfo1)
PRINT_DEPINFO1(nanos6_register_region_readwrite_depinfo1)

#undef PRINT_DEPINFO1

#ifdef __cplusplus
}
#endif
