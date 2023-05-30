// RUN: %clang_cc1 -fompss-2 -verify -DFOMPSS_2 -o - %s
// RUN: %clang_cc1 -verify -o - %s

// expected-no-diagnostics
#ifdef FOMPSS_2
// -fompss-2 option is specified
#ifndef _OMPSS_2
#error "No _OMPSS_2 macro is defined with -fompss-2 option"
#elif _OMPSS_2 != 1
#error "_OMPSS_2 has incorrect value"
#endif //_OMPSS_2
#ifndef _OMPSS_2_NANOS6
#error "No _OMPSS_2_NANOS6 macro is defined with -fompss-2=libnanos6 option"
#endif
#else
// No -fompss-2 option is specified
#ifdef _OMPSS_2
#error "_OMPSS_2 macro is defined without -fompss-2 option"
#endif // _OMPSS_2
#endif // FOMPSS_2

