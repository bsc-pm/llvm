// RUN: %clang -target x86_64-linux-gnu -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-apple-darwin -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-freebsd -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
// RUN: %clang -target x86_64-netbsd -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OMPSS
//
// CHECK-CC1-OMPSS: "-cc1"
// CHECK-CC1-OMPSS: "-fompss-2"

// -fdo-not-use-ompss-2-runtime does not need to be passed to frontend

// RUN: %clang -target x86_64-linux-gnu -fdo-not-use-ompss-2-runtime -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-NANOS6
// RUN: %clang -target x86_64-apple-darwin -fdo-not-use-ompss-2-runtime -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-NANOS6
// RUN: %clang -target x86_64-freebsd -fdo-not-use-ompss-2-runtime -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-NANOS6
// RUN: %clang -target x86_64-netbsd -fdo-not-use-ompss-2-runtime -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-NANOS6
//
// CHECK-CC1-NO-NANOS6: "-cc1"
// CHECK-CC1-NO-NANOS6-NOT: "-fdo-not-use-ompss-2-runtime"
