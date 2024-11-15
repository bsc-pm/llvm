#ifndef HSEARCH_H
#define HSEARCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

typedef union htab_value {
	void *p;
	size_t n;
} htab_value;

#define HTV_N(N) (htab_value) {.n = N}
#define HTV_P(P) (htab_value) {.p = P}

struct htab * htab_create(size_t);
void htab_destroy(struct htab *);
htab_value* htab_find(struct htab *, char* key);
/* same as htab_find, but can retrieve the saved key (for freeing) */
htab_value* htab_find2(struct htab *htab, char* key, char **saved_key);
int htab_insert(struct htab *, char*, htab_value);
int htab_delete(struct htab *htab, char* key);
size_t htab_next(struct htab *, size_t iterator, char** key, htab_value **v);

#endif

#ifdef __cplusplus
}
#endif
