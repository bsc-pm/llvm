! RUN: flang-new -fc1 -flang-deprecated-no-hlfir -fompss-2 -emit-llvm  -mmlir --mlir-print-ir-after-all %s -o /dev/null |& tail -n +$(flang-new -fc1 -flang-deprecated-no-hlfir -fompss-2 -emit-llvm  -mmlir --mlir-print-ir-after-all %s -o /dev/null |& grep -n FIRToLLVMLowering | cut -f1 -d:) | FileCheck %s --check-prefix=LLVMIRDialect

PROGRAM P
INTEGER :: X
INTEGER :: Y
INTEGER :: ARRAY(10, 10)

!$OSS TASK
DO X=0,10
DO Y=0,10
if (Y .eq. 7) EXIT
ARRAY(X,Y) = X + Y
END DO
END DO
!$OSS END TASK

END PROGRAM

! This tests checks allocas inside the directive
! and unstructured code generation.

!LLVMIRDialect: oss.task
!LLVMIRDialect:   %{{.*}} = llvm.alloca
!LLVMIRDialect:   %{{.*}} = llvm.alloca
!LLVMIRDialect:   llvm.br ^[[BB1:.*]]
!LLVMIRDialect: ^[[BB1]]:
!LLVMIRDialect:   llvm.cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB7:.*]]
!LLVMIRDialect: ^[[BB2]]:
!LLVMIRDialect:   llvm.br ^[[BB3:.*]]
!LLVMIRDialect: ^[[BB3]]:
!LLVMIRDialect:   llvm.cond_br %{{.*}}, ^[[BB4:.*]], ^[[BB6:.*]]
!LLVMIRDialect: ^[[BB4]]:
!LLVMIRDialect:   llvm.cond_br %{{.*}}, ^[[BB6]], ^[[BB5:.*]]
!LLVMIRDialect: ^[[BB5]]:
!LLVMIRDialect:   llvm.br ^bb3
!LLVMIRDialect: ^[[BB6]]:
!LLVMIRDialect:   llvm.br ^bb1
!LLVMIRDialect: ^[[BB7]]:
!LLVMIRDialect:   oss.terminator
