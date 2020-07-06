// RUN: %clang %s -c -E -dM -fompss-2 | FileCheck --check-prefix=USING-OMPSS-2 %s
// RUN: %clang %s -c -E -dM | FileCheck --check-prefix=NO-USING-OMPSS-2 %s

// USING-OMPSS-2: #define _OMPSS_2 1
// NO-USING-OMPSS-2-NOT: #define _OMPSS_2
