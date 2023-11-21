! RUN: bbc -hlfir=false -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

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
