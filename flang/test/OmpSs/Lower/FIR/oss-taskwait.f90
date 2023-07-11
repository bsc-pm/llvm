! This test checks lowering of OmpSs-2 taskwait Directive.

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program taskwait
    INTEGER :: I

    !$OSS TASKWAIT
    !$OSS TASKWAIT DEPEND(IN: I)
    !$OSS TASKWAIT IN(I)

end program


!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!FIRDialect-NEXT:  oss.taskwait
!FIRDialect-NEXT:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I]] : !fir.ref<i32>) function(@compute.dep0) arguments(%[[VAR_I]] : !fir.ref<i32>) -> i32
!FIRDialect-NEXT:  %[[FALSE:.*]] = arith.constant false
!FIRDialect-NEXT:  oss.task
!FIRDialect-SAME:  if(%[[FALSE]] : i1)
!FIRDialect-SAME:  shared(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME:  in(%[[DEP_0]] : i32)
!FIRDialect:  %[[DEP_1:.*]] = oss.dependency base(%[[VAR_I]] : !fir.ref<i32>) function(@compute.dep1) arguments(%[[VAR_I]] : !fir.ref<i32>) -> i32
!FIRDialect-NEXT:  %[[FALSE_0:.*]] = arith.constant false
!FIRDialect-NEXT:  oss.task if(%[[FALSE_0]] : i1)
!FIRDialect-SAME:  shared(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME:  in(%[[DEP_1]] : i32)
