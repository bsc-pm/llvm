! RUN: %flang -target x86_64-linux-gnu -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OMPSS2
! RUN: %flang -target x86_64-apple-darwin -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OMPSS2
! RUN: %flang -target x86_64-freebsd -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OMPSS2
! RUN: %flang -target x86_64-windows-gnu -fompss-2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OMPSS2

! CHECK-FC1-OMPSS2: "-fc1"
! CHECK-FC1-OMPSS2: "-fompss-2"
!
