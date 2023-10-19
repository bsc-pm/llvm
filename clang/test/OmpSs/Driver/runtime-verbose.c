// RUN: NANOS6_HOME=ASDF %clang -fompss-2=libnanos6 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NANOS6
// RUN: NODES_HOME=FDSA %clang -fompss-2=libnodes %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NODES

// CHECK-NANOS6: clang version{{.*}}
// CHECK-NANOS6: "{{[^"]*}}clang{{[^"]*}}"
// CHECK-NANOS6-SAME: "-I" "ASDF/include"
// CHECK-NANOS6: "{{[^"]*}}ld{{(.exe)?}}"
// CHECK-NANOS6-SAME: "ASDF/lib/nanos6-main-wrapper.o"

// Ensure nosv is placed before nodes
// FIXME? Should we put nosv at the beginning of the whole
// line?
// CHECK-NODES: clang version{{.*}}
// CHECK-NODES: "{{[^"]*}}clang{{[^"]*}}"
// CHECK-NODES-SAME: "-I" "FDSA/include"
// CHECK-NODES: "{{[^"]*}}ld{{(.exe)?}}"
// CHECK-NODES-SAME: "-lnosv"
// CHECK-NODES-SAME: "FDSA/lib/nodes-main-wrapper.o"
// CHECK-NODES-SAME: "-lnodes"

