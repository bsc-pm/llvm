! This test checks lowering of OmpSs-2 DepOp for multidimensional arrays.

! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program task
    INTEGER :: ARRAY(10, 10)

    !$OSS TASK IN(ARRAY)
    !$OSS END TASK
end

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_ARRAY:.*]] = fir.address_of(@_QFEarray) : !fir.ref<!fir.array<10x10xi32>>
!FIRDialect:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10x10xi32>>) function(@compute.dep0) arguments(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10x10xi32>>) -> i32
!FIRDialect-NEXT:  oss.task
!FIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10x10xi32>>)
!FIRDialect-SAME:  captures(%{{.*}}, %{{.*}} : index, index)
!FIRDialect-SAME:  in(%[[DEP_0]] : i32)

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: !fir.ref<!fir.array<10x10xi32>>) -> (!fir.ref<!fir.array<10x10xi32>>, i64, i64, i64, i64, i64, i64)
!FIRDialect:   %c10 = arith.constant 10 : index
!FIRDialect-NEXT:   %c10_0 = arith.constant 10 : index
!FIRDialect-NEXT:   %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c10_i64_1 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c10_i64_2 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c1_i64_3 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c10_i64_4 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c1_i64_5 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:   %1 = arith.subi %c10_i64_1, %c1_i64 : i64
!FIRDialect-NEXT:   %2 = arith.addi %1, %c1_i64_5 : i64
!FIRDialect-NEXT:   %3 = arith.subi %c1_i64_3, %c1_i64_3 : i64
!FIRDialect-NEXT:   %4 = arith.subi %c10_i64_4, %c1_i64_3 : i64
!FIRDialect-NEXT:   %5 = arith.addi %4, %c1_i64_5 : i64
!FIRDialect-NEXT:   %6 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:   %7 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT:   %8 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:   return %arg0, %6, %7, %8, %c10_i64_2, %3, %5 : !fir.ref<!fir.array<10x10xi32>>, i64, i64, i64, i64, i64, i64

