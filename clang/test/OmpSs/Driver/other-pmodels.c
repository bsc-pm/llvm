// RUN: %clang -fompss-2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-OMPSS2
// RUN: %clang -fopenmp=libompv %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-OPENMP

// CHECK-OPENMP: clang version{{.*}}
// CHECK-OPENMP: "{{[^"]*}}clang{{[^"]*}}"
// CHECK-OPENMP-SAME: "-Wsource-uses-ompss-2"

// CHECK-OMPSS2: clang version{{.*}}
// CHECK-OMPSS2: "{{[^"]*}}clang{{[^"]*}}"
// CHECK-OMPSS2-SAME: "-Wsource-uses-openmp"
