// RUN: %clang -target x86_64-linux-gnu -fompss -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-apple-darwin -fompss -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-freebsd -fompss -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-netbsd -fompss -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
//
// CHECK-CC1-OMPSS: "-cc1"
// CHECK-CC1-OMPSS: "-fompss"
//
// CHECK-CC1-NO-OMPSS: "-cc1"
// CHECK-CC1-NO-OMPSS-NOT: "-fompss"

