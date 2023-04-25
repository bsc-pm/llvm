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
