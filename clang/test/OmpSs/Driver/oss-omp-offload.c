// RUN: %clang -fompss-2 -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-OMPSS2
//
// Verify we introduce -fompss-2-target so OmpSs-2 Transform pass is run in the -disable-llvm-passes action
// CHECK-OMPSS2: clang version{{.*}}
// CHECK-OMPSS2: "{{[^"]*}}clang{{[^"]*}}"
// CHECK-OMPSS2-SAME: "-fompss-2"
// CHECK-OMPSS2-SAME: "-disable-llvm-passes"
// CHECK-OMPSS2-SAME: "-fopenmp-targets=x86_64-pc-linux-gnu"
// CHECK-OMPSS2-SAME: "-fompss-2-target"



