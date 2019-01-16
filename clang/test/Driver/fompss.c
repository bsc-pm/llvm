// RUN: %clang -target x86_64-linux-gnu -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-apple-darwin -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-freebsd -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-netbsd -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
//
// CHECK-CC1-OMPSS: "-cc1"
// CHECK-CC1-OMPSS: "-fompss-2"
//
// CHECK-CC1-NO-OMPSS: "-cc1"
// CHECK-CC1-NO-OMPSS-NOT: "-fompss-2"

