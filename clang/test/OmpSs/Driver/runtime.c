// CompileJob
// RUN: %clang -fompss-2=libnanos6 %s -c -### 2>&1
// RUN: %clang -fompss-2=libnodes %s -c -### 2>&1
// RUN: %clang -fompss-2=libasdf %s -c -### 2>&1 | FileCheck %s --check-prefix=CHECK1
// LinkJob
// RUN: touch %t.o
// RUN: %clang -fompss-2=libnanos6 %t.o -### 2>&1
// RUN: %clang -fompss-2=libnodes %t.o -### 2>&1
// RUN: %clang -fompss-2=libasdf %t.o -### 2>&1 | FileCheck %s --check-prefix=CHECK2
// CompileAndLinkJob
// RUN: %clang -fompss-2=libnanos6 %s -### 2>&1
// RUN: %clang -fompss-2=libnodes %s -### 2>&1
// RUN: %clang -fompss-2=libasdf %s -### 2>&1 | FileCheck %s --check-prefix=CHECK3

// CHECK1: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
// CHECK2: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
// CHECK3: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
// CHECK3: clang: error: unsupported argument 'libasdf' to option '-fompss-2='

// OMPSS2_RUNTIME priority check

// CompileJob
// RUN: OMPSS2_RUNTIME=libkaka %clang -fompss-2 %s -c -### 2>&1 | FileCheck %s --check-prefix=CHECK4
// RUN: OMPSS2_RUNTIME=libkaka %clang -fompss-2=libasdf %s -c -### 2>&1 | FileCheck %s --check-prefix=CHECK5
// LinkJob
// RUN: touch %t.o
// RUN: OMPSS2_RUNTIME=libkaka %clang -fompss-2 %t.o -### 2>&1 | FileCheck %s --check-prefix=CHECK6
// RUN: OMPSS2_RUNTIME=libkaka %clang -fompss-2=libasdf %t.o -### 2>&1 | FileCheck %s --check-prefix=CHECK7
// CompileAndLinkJob
// RUN: OMPSS2_RUNTIME=libkaka %clang -fompss-2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK8
// RUN: OMPSS2_RUNTIME=libkaka %clang -fompss-2=libasdf %s -### 2>&1 | FileCheck %s --check-prefix=CHECK9

// CHECK4: clang: error: unsupported argument 'libkaka' to option '-fompss-2='
// CHECK5: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
// CHECK6: clang: error: unsupported argument 'libkaka' to option '-fompss-2='
// CHECK7: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
// CHECK8: clang: error: unsupported argument 'libkaka' to option '-fompss-2='
// CHECK8: clang: error: unsupported argument 'libkaka' to option '-fompss-2='
// CHECK9: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
// CHECK9: clang: error: unsupported argument 'libasdf' to option '-fompss-2='
