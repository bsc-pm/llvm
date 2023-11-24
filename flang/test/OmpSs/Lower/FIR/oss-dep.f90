! This test checks lowering of OmpSs-2 DepOp.

! Test for program

! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

! Support list
! - [x] scalar
! - [x] explicit-shape array (no region)
! - [x] explicit-shape array (region)
! - [ ] deferred-shape array
! - [ ] assumed-size array
! - [ ] assumed-shape array

program task
    INTEGER :: J

    INTEGER :: I
    INTEGER :: ARRAY(10)
    INTEGER :: ARRAY1(5:10)
    TYPE TY
       INTEGER :: X
       INTEGER :: ARRAY(10)
    END TYPE
    TYPE(TY) T

    !$OSS TASK DO DEPEND(OUT: I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: T%X)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: T%ARRAY)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: T%ARRAY(2))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY1)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I : I + 10))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(:))
    DO J=1,10
    END DO
    !$OSS END TASK DO

end program

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_ARRAY:.*]] = fir.address_of(@_QFEarray) : !fir.ref<!fir.array<10xi32>>
!FIRDialect:  %[[VAR_ARRAY1:.*]] = fir.alloca !fir.array<6xi32> {bindc_name = "array1", uniq_name = "_QFEarray1"}
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!FIRDialect:  %[[VAR_T:.*]] = fir.address_of(@_QFEt) : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>
!FIRDialect:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I]] : !fir.ref<i32>) function(@compute.dep0) arguments(%[[VAR_I]] : !fir.ref<i32>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  shared(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME:  out(%[[DEP_0]] : i32)

!FIRDialect:  %[[DEP_1:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>) function(@compute.dep1) arguments(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>)
!FIRDialect-SAME:  out(%[[DEP_1]] : i32)

!FIRDialect:  %[[DEP_2:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>) function(@compute.dep2) arguments(%[[VAR_ARRAY]], %[[VAR_I]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<i32>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>)
!FIRDialect-SAME:  out(%[[DEP_2]] : i32)

!FIRDialect:  %[[DEP_3:.*]] = oss.dependency base(%[[VAR_T]] : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>) function(@compute.dep3) arguments(%[[VAR_T]] : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  shared(%[[VAR_T]] : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>)
!FIRDialect-SAME:  out(%[[DEP_3]] : i32)

!FIRDialect:  %[[DEP_4:.*]] = oss.dependency base(%[[VAR_T]] : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>) function(@compute.dep4) arguments(%[[VAR_T]] : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  shared(%[[VAR_T]] : !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>)
!FIRDialect-SAME:  out(%[[DEP_4]] : i32)

!FIRDialect:  %[[DEP_5:.*]] = oss.dependency base(%[[VAR_ARRAY1]] : !fir.ref<!fir.array<6xi32>>) function(@compute.dep5) arguments(%[[VAR_ARRAY1]] : !fir.ref<!fir.array<6xi32>>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  shared(%[[VAR_ARRAY1]] : !fir.ref<!fir.array<6xi32>>)
!FIRDialect-SAME:  captures(%{{.*}}, %{{.*}} : index, index)
!FIRDialect-SAME:  out(%[[DEP_5]] : i32)

!FIRDialect:  %[[DEP_6:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>) function(@compute.dep6) arguments(%[[VAR_ARRAY]], %[[VAR_I]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<i32>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>)
!FIRDialect-SAME:  captures(%{{.*}} : index)
!FIRDialect-SAME:  out(%[[DEP_6]] : i32)

!FIRDialect:  %[[DEP_7:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>) function(@compute.dep7) arguments(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>) -> i32
!FIRDialect:  oss.task_for
!FIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !fir.ref<!fir.array<10xi32>>)
!FIRDialect-SAME:  out(%[[DEP_7]] : i32)

!FIRDialect-LABEL: func @compute.dep0(%arg0: !fir.ref<i32>) -> (!fir.ref<i32>, i64, i64, i64)
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:  %1 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:  %2 = arith.addi %1, %c1_i64_0 : i64
!FIRDialect-NEXT:  %3 = arith.muli %c1_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT:  %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %3, %4, %5 : !fir.ref<i32>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep1(%arg0: !fir.ref<!fir.array<10xi32>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:  %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c10_i64_0 = arith.constant 10 : i64
!FIRDialect-NEXT:  %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:  %1 = arith.subi %c10_i64_0, %c1_i64 : i64
!FIRDialect-NEXT:  %2 = arith.addi %1, %c1_i64_1 : i64
!FIRDialect-NEXT:  %3 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT:  %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %3, %4, %5 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep2(%arg0: !fir.ref<!fir.array<10xi32>>, %arg1: !fir.ref<i32>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:  %1 = fir.convert %0 : (i32) -> i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %2 = arith.subi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %3 = arith.subi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %4 = arith.addi %3, %c1_i64_0 : i64
!FIRDialect-NEXT:  %5 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %6 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  %7 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %5, %6, %7 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep3(%arg0: !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>) -> (!fir.ref<i32>, i64, i64, i64)
!FIRDialect:  %0 = fir.field_index x, !fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>
!FIRDialect-NEXT:  %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>, !fir.field) -> !fir.ref<i32>
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %2 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:  %3 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:  %4 = arith.addi %3, %c1_i64_0 : i64
!FIRDialect-NEXT:  %5 = arith.muli %c1_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %6 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  %7 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:  return %1, %5, %6, %7 : !fir.ref<i32>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep4(%arg0: !fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.field_index array, !fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>
!FIRDialect-NEXT:   %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QFTty{x:i32,array:!fir.array<10xi32>}>>, !fir.field) -> !fir.ref<!fir.array<10xi32>>
!FIRDialect-NEXT:   %c10 = arith.constant 10 : index
!FIRDialect-NEXT:   %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c10_i64_0 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %2 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:   %3 = arith.subi %c10_i64_0, %c1_i64 : i64
!FIRDialect-NEXT:   %4 = arith.addi %3, %c1_i64_1 : i64
!FIRDialect-NEXT:   %5 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:   %6 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:   %7 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:   return %1, %5, %6, %7 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep5(%arg0: !fir.ref<!fir.array<6xi32>>) -> (!fir.ref<!fir.array<6xi32>>, i64, i64, i64)
!FIRDialect:  %c6_i64 = arith.constant 6 : i64
!FIRDialect-NEXT:  %c5_i64 = arith.constant 5 : i64
!FIRDialect-NEXT:  %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = arith.subi %c5_i64, %c5_i64 : i64
!FIRDialect-NEXT:  %1 = arith.subi %c10_i64, %c5_i64 : i64
!FIRDialect-NEXT:  %2 = arith.addi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %3 = arith.muli %c6_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT:  %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %3, %4, %5 : !fir.ref<!fir.array<6xi32>>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep6(%arg0: !fir.ref<!fir.array<10xi32>>, %arg1: !fir.ref<i32>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:  %1 = fir.convert %0 : (i32) -> i64
!FIRDialect-NEXT:  %2 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:  %c10_i32 = arith.constant 10 : i32
!FIRDialect-NEXT:  %3 = arith.addi %2, %c10_i32 : i32
!FIRDialect-NEXT:  %4 = fir.convert %3 : (i32) -> i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %5 = arith.subi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %6 = arith.subi %4, %c1_i64 : i64
!FIRDialect-NEXT:  %7 = arith.addi %6, %c1_i64_0 : i64
!FIRDialect-NEXT:  %8 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %9 = arith.muli %5, %c4_i64 : i64
!FIRDialect-NEXT:  %10 = arith.muli %7, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %8, %9, %10 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep7(%arg0: !fir.ref<!fir.array<10xi32>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:  %c10_i64_0 = arith.constant 10 : i64
!FIRDialect-NEXT:  %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT:  %1 = arith.subi %c10_i64_0, %c1_i64 : i64
!FIRDialect-NEXT:  %2 = arith.addi %1, %c1_i64_1 : i64
!FIRDialect-NEXT:  %3 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT:  %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %3, %4, %5 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

