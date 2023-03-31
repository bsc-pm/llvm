// RUN: clang-import-test --Xcc=-fompss-2 -dump-ast -import %S/Inputs/attributes.cpp -expression %s | FileCheck %s

void expr() {
    f();
}
// CHECK: OSSTaskDeclAttr 