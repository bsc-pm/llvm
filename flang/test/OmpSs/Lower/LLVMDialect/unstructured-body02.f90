! RUN: flang-new -fc1 -fompss-2 -emit-llvm -flang-deprecated-no-hlfir -fdisable-ompss-2-pass -mmlir --mlir-print-ir-after-all %s -o /dev/null |& tail -n +$(flang-new -fc1 -fompss-2 -emit-llvm -flang-deprecated-no-hlfir -fdisable-ompss-2-pass -mmlir --mlir-print-ir-after-all %s -o /dev/null |& grep -n FIRToLLVMLowering | cut -f1 -d:) | FileCheck %s --check-prefix=LLVMIRDialect

PROGRAM P
INTEGER :: X
INTEGER :: Y
INTEGER :: ARRAY(10, 10)

!$OSS TASK DO
DO X=0,10
DO Y=0,10
if (Y .eq. 7) EXIT
ARRAY(X,Y) = X + Y
!Y = Y + 1
END DO
END DO
!$OSS END TASK DO

END PROGRAM

! This tests checks allocas inside the directive
! and unstructured code generation.

!LLVMIRDialect: oss.task_for
!LLVMIRDialect:   %{{.*}} = llvm.alloca
!LLVMIRDialect:   llvm.br ^[[BB1:.*]]
!LLVMIRDialect: ^[[BB1]]:
!LLVMIRDialect:   llvm.cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB4:.*]]
!LLVMIRDialect: ^[[BB2]]:
!LLVMIRDialect:   llvm.cond_br %{{.*}}, ^[[BB4]], ^[[BB3:.*]]
!LLVMIRDialect: ^[[BB3]]:
!LLVMIRDialect:   llvm.br ^[[BB1]]
!LLVMIRDialect: ^[[BB4]]:
!LLVMIRDialect:   oss.terminator
