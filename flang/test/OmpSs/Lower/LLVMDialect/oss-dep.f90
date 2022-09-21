! This test checks lowering of OmpSs-2 DepOp.

! Test for program

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

! Support list
! - [x] scalar
! - [x] explicit-shape array (no region)
! - [x] explicit-shape array (region)
! - [ ] assumed-size array
! - [ ] assumed-shape array
! - [ ] deferred-shape array

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

    ! TODO
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

!LLVMIRDialect-LABEL: llvm.func @_QQmain()
!LLVMIRDialect:  %[[CONSTANT_10:.*]] = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect:  %[[VAR_ARRAY:.*]] = llvm.mlir.addressof @_QFEarray : !llvm.ptr<array<10 x i32>>
!LLVMIRDialect-NEXT:  %[[VAR_ARRAY1:.*]] = llvm.mlir.addressof @_QFEarray1 : !llvm.ptr<array<6 x i32>>
!LLVMIRDialect-NEXT:  %[[CONSTANT_1_0:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_1_0]]
!LLVMIRDialect-NEXT:  %[[CONSTANT_1_1:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect:  %[[CONSTANT_1_2:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_T:.*]] = llvm.alloca %[[CONSTANT_1_2]]

!LLVMIRDialect-NEXT:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I]] : !llvm.ptr<i32>) function(@compute.dep0) arguments(%[[VAR_I]] : !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_I]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  out(%[[DEP_0]] : i32)

!LLVMIRDialect:  %[[DEP_1:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>) function(@compute.dep1) arguments(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>)
!LLVMIRDialect-SAME:  out(%[[DEP_1]] : i32)

!LLVMIRDialect:  %[[DEP_2:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>) function(@compute.dep2) arguments(%[[VAR_ARRAY]], %[[VAR_I]] : !llvm.ptr<array<10 x i32>>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>)
!LLVMIRDialect-SAME:  out(%[[DEP_2]] : i32)

!LLVMIRDialect:  %[[DEP_3:.*]] = oss.dependency base(%[[VAR_T]] : !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) function(@compute.dep3) arguments(%[[VAR_T]] : !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_T]] : !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>)
!LLVMIRDialect-SAME:  out(%[[DEP_3]] : i32)

!LLVMIRDialect:  %[[DEP_4:.*]] = oss.dependency base(%[[VAR_T]] : !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) function(@compute.dep4) arguments(%[[VAR_T]] : !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_T]] : !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>)
!LLVMIRDialect-SAME:  out(%[[DEP_4]] : i32)

!LLVMIRDialect:  %[[DEP_5:.*]] = oss.dependency base(%[[VAR_ARRAY1]] : !llvm.ptr<array<6 x i32>>) function(@compute.dep5) arguments(%[[VAR_ARRAY1]] : !llvm.ptr<array<6 x i32>>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY1]] : !llvm.ptr<array<6 x i32>>)
!LLVMIRDialect-SAME:  out(%[[DEP_5]] : i32)

!LLVMIRDialect:  %[[DEP_6:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>) function(@compute.dep6) arguments(%[[VAR_ARRAY]], %[[VAR_I]] : !llvm.ptr<array<10 x i32>>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>)
!LLVMIRDialect-SAME:  out(%[[DEP_6]] : i32)

!LLVMIRDialect:  %[[DEP_7:.*]] = oss.dependency base(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>) function(@compute.dep7) arguments(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY]] : !llvm.ptr<array<10 x i32>>)
!LLVMIRDialect-SAME:  out(%[[DEP_7]] : i32)

!LLVMIRDialect-LABEL: llvm.func @compute.dep0(%arg0: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:  %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:  %1 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:  %2 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %6 = llvm.insertvalue %1, %5[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  llvm.return %6 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep1(%arg0: !llvm.ptr<array<10 x i32>>) -> !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect:  %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:  %1 = llvm.mlir.constant(40 : i64) : i64
!LLVMIRDialect-NEXT:  %2 = llvm.mlir.undef : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  %6 = llvm.insertvalue %1, %5[3] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:  llvm.return %6 : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep2(%arg0: !llvm.ptr<array<10 x i32>>, %arg1: !llvm.ptr<i32>) -> !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(40 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:   %3 = llvm.load %arg1 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %4 = llvm.sext %3 : i32 to i64
!LLVMIRDialect-NEXT:   %5 = llvm.sub %4, %2  : i64
!LLVMIRDialect-NEXT:   %6 = llvm.mul %5, %1  : i64
!LLVMIRDialect-NEXT:   %7 = llvm.mul %4, %1  : i64
!LLVMIRDialect-NEXT:   %8 = llvm.mlir.undef : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %9 = llvm.insertvalue %arg0, %8[0] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %10 = llvm.insertvalue %0, %9[1] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %11 = llvm.insertvalue %6, %10[2] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %12 = llvm.insertvalue %7, %11[3] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %12 : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep3(%arg0: !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(0 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) -> !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %4 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %5 = llvm.insertvalue %3, %4[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %6 = llvm.insertvalue %1, %5[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %7 = llvm.insertvalue %0, %6[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %8 = llvm.insertvalue %1, %7[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %8 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep4(%arg0: !llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) -> !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(40 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(1 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr<struct<"_QFTty", (i32, array<10 x i32>)>>) -> !llvm.ptr<array<10 x i32>>
!LLVMIRDialect-NEXT:   %4 = llvm.mlir.undef : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %5 = llvm.insertvalue %3, %4[0] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %6 = llvm.insertvalue %1, %5[1] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %7 = llvm.insertvalue %0, %6[2] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %8 = llvm.insertvalue %1, %7[3] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %8 : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep5(%arg0: !llvm.ptr<array<6 x i32>>) -> !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(24 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.undef : !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %6 = llvm.insertvalue %1, %5[3] : !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %6 : !llvm.struct<(ptr<array<6 x i32>>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep6(%arg0: !llvm.ptr<array<10 x i32>>, %arg1: !llvm.ptr<i32>) -> !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(40 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:   %4 = llvm.load %arg1 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %5 = llvm.sext %4 : i32 to i64
!LLVMIRDialect-NEXT:   %6 = llvm.add %4, %2  : i32
!LLVMIRDialect-NEXT:   %7 = llvm.sext %6 : i32 to i64
!LLVMIRDialect-NEXT:   %8 = llvm.sub %5, %3  : i64
!LLVMIRDialect-NEXT:   %9 = llvm.mul %8, %1  : i64
!LLVMIRDialect-NEXT:   %10 = llvm.mul %7, %1  : i64
!LLVMIRDialect-NEXT:   %11 = llvm.mlir.undef : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %12 = llvm.insertvalue %arg0, %11[0] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %13 = llvm.insertvalue %0, %12[1] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %14 = llvm.insertvalue %9, %13[2] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %15 = llvm.insertvalue %10, %14[3] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %15 : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep7(%arg0: !llvm.ptr<array<10 x i32>>) -> !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(40 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.undef : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %6 = llvm.insertvalue %1, %5[3] : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %6 : !llvm.struct<(ptr<array<10 x i32>>, i64, i64, i64)>

