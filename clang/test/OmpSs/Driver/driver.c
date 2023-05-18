// RUN: %clang %s -c -E -dM -fompss-2 | FileCheck --check-prefix=USING-OMPSS-2 %s
// RUN: %clang %s -c -E -dM | FileCheck --check-prefix=NO-USING-OMPSS-2 %s

// RUN: %clang %s -c -E -dM -fompss-2=libnodes | FileCheck --check-prefix=USING-OMPSS-2-NODES %s
// RUN: %clang %s -c -E -dM | FileCheck --check-prefix=NO-USING-OMPSS-2-NODES %s

// RUN: %clang %s -c -E -dM -fompss-2=libnanos6 | FileCheck --check-prefix=USING-OMPSS-2-NANOS6 %s
// RUN: %clang %s -c -E -dM | FileCheck --check-prefix=NO-USING-OMPSS-2-NANOS6 %s


// USING-OMPSS-2: #define _OMPSS_2 1
// NO-USING-OMPSS-2-NOT: #define _OMPSS_2

// USING-OMPSS-2-NODES: #define _OMPSS_2
// USING-OMPSS-2-NODES: #define _OMPSS_2_NODES 1
// NO-USING-OMPSS-2-NODES-NOT: #define _OMPSS_2
// NO-USING-OMPSS-2-NODES-NOT: #define _OMPSS_2_NODES

// USING-OMPSS-2-NANOS6: #define _OMPSS_2
// USING-OMPSS-2-NANOS6: #define _OMPSS_2_NANOS6 1
// NO-USING-OMPSS-2-NANOS6-NOT: #define _OMPSS_2
// NO-USING-OMPSS-2-NANOS6-NOT: #define _OMPSS_2_NANOS6
