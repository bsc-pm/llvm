// RUN: clang-import-test --Xcc=-fompss-2 -dump-ast -import %S/Inputs/clauses.cpp -expression %s | FileCheck %s

void expr() {
    f(nullptr);
}

// task attr directive (cuda)
// CHECK: OSSTaskDeclAttr
// CHECK-SAME: Cuda
// CHECK: DeclRefExpr
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: IntegerLiteral

// Task directive
// CHECK: OSSIfClause
// CHECK: OSSFinalClause
// CHECK: OSSCostClause
// CHECK: OSSPriorityClause
// CHECK: OSSLabelClause
// CHECK: OSSOnreadyClause
// CHECK: OSSWaitClause
// CHECK: OSSDefaultClause
// CHECK: OSSDependClause
// CHECK: OSSArraySectionExpr
// CHECK: OSSMultiDepExpr
// CHECK: OSSReductionClause
// CHECK: OSSSharedClause
// CHECK: OSSPrivateClause
// CHECK: OSSFirstprivateClause

// atomic directive
// CHECK: OSSUpdateClause
// CHECK: OSSReadClause
// CHECK: OSSWriteClause
// CHECK: OSSSeq_cstClause
// CHECK: OSSReleaseClause
// CHECK: OSSRelaxedClause


// loop directive
// CHECK: OSSChunksizeClause
// CHECK: OSSGrainsizeClause
// CHECK: OSSCollapseClause
// CHECK: OSSUnrollClause

// task attr directive
// CHECK: OSSTaskDeclAttr
// CHECK-SAME: CopyDeps
// CHECK-SAME: Fpga
// These just print their expression
// CHECK: 1337
// CHECK: 12884901888
// CHECK: 1234
// CHECK: 1000
// CHECK: 1234567
// CHECK: OSSArrayShapingExpr
// CHECK-SAME: int[100]
// CHECK: OSSArrayShapingExpr
// CHECK-SAME: int[100]
// CHECK: OSSArrayShapingExpr
// CHECK-SAME: int[100]
