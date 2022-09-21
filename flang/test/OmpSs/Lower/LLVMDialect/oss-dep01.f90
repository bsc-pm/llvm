! This test checks lowering of OmpSs-2 DepOp.

! Test for subroutine

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

! Support list
! - [x] assumed-size array
! - [x] assumed-shape array

subroutine task(X, ARRAY, ARRAY1)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(*)
    INTEGER :: ARRAY1(:)
    INTEGER :: ARRAY2(4: x + 6)

    INTEGER :: I
    INTEGER :: J

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1)
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY2)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1(I))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY2(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I : I + 10))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1(I : I + 10))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    ! TODO
    !$OSS TASK DO DEPEND(OUT: ARRAY2( : ))
    DO J=1,10
    END DO
    !$OSS END TASK DO

end subroutine

!LLVMIRDialect-LABEL: llvm.func @_QPtask(%arg0: !llvm.ptr<i32> {fir.bindc_name = "x"}, %arg1: !llvm.ptr<i32> {fir.bindc_name = "array"}, %arg2: !llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>> {fir.bindc_name = "array1"}) {
!LLVMIRDialect:  %[[VAR_I:.*]] = llvm.alloca
!LLVMIRDialect:  %[[VAR_J:.*]] = llvm.alloca
!LLVMIRDialect:  %[[VAR_ARRAY2:.*]] = llvm.alloca
!LLVMIRDialect:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !llvm.ptr<i32>) function(@compute.dep0) arguments(%arg0, %[[VAR_ARRAY2]] : !llvm.ptr<i32>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY2]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  out(%[[DEP_0]] : i32)

!LLVMIRDialect:  %[[DEP_1:.*]] = oss.dependency base(%arg1 : !llvm.ptr<i32>) function(@compute.dep1) arguments(%arg1, %[[VAR_I]] : !llvm.ptr<i32>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  shared(%arg1 : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  out(%[[DEP_1]] : i32)

!LLVMIRDialect:  %[[DEP_2:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !llvm.ptr<i32>) function(@compute.dep2) arguments(%arg0, %[[VAR_ARRAY2]], %[[VAR_I]] : !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY2]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  out(%[[DEP_2]] : i32)

!LLVMIRDialect:  %[[DEP_3:.*]] = oss.dependency base(%arg1 : !llvm.ptr<i32>) function(@compute.dep3) arguments(%arg1, %[[VAR_I]] : !llvm.ptr<i32>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  shared(%arg1 : !llvm.ptr<i32>)
!LLVMIRDialect-SMAE:  out(%[[DEP_3]] : i32)

!LLVMIRDialect:  %[[DEP_4:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !llvm.ptr<i32>) function(@compute.dep4) arguments(%arg0, %[[VAR_ARRAY2]] : !llvm.ptr<i32>, !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT:  oss.task_for
!LLVMIRDialect-SAME:  shared(%[[VAR_ARRAY2]] : !llvm.ptr<i32>)
!LLVMIRDialect-SAME:  out(%[[DEP_4]] : i32)

!LLVMIRDialect-LABEL: llvm.func @compute.dep0(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(-3 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(6 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %4 = llvm.load %arg0 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %5 = llvm.add %4, %2  : i32
!LLVMIRDialect-NEXT:   %6 = llvm.sext %5 : i32 to i64
!LLVMIRDialect-NEXT:   %7 = llvm.add %6, %0  : i64
!LLVMIRDialect-NEXT:   %8 = llvm.mul %7, %3  : i64
!LLVMIRDialect-NEXT:   %9 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %11 = llvm.insertvalue %8, %10[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %12 = llvm.insertvalue %1, %11[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %13 = llvm.insertvalue %8, %12[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %13 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

!LLVMIRDialect-LABEL llvm.func @compute.dep1(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.load %arg1 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %3 = llvm.sext %2 : i32 to i64
!LLVMIRDialect-NEXT:   %4 = llvm.sub %3, %1  : i64
!LLVMIRDialect-NEXT:   %5 = llvm.mul %4, %0  : i64
!LLVMIRDialect-NEXT:   %6 = llvm.mul %3, %0  : i64
!LLVMIRDialect-NEXT:   %7 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %8 = llvm.insertvalue %arg0, %7[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %9 = llvm.insertvalue %0, %8[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %10 = llvm.insertvalue %5, %9[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %11 = llvm.insertvalue %6, %10[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   llvm.return %11 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep2(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(-3 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(6 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %4 = llvm.load %arg0 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %5 = llvm.add %4, %2  : i32
!LLVMIRDialect-NEXT:   %6 = llvm.sext %5 : i32 to i64
!LLVMIRDialect-NEXT:   %7 = llvm.add %6, %0  : i64
!LLVMIRDialect-NEXT:   %8 = llvm.icmp "sgt" %7, %1 : i64
!LLVMIRDialect-NEXT:   %9 = llvm.select %8, %7, %1 : i1, i64
!LLVMIRDialect-NEXT:   %10 = llvm.load %arg2 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %11 = llvm.sext %10 : i32 to i64
!LLVMIRDialect-NEXT:   %12 = llvm.sub %11, %3  : i64
!LLVMIRDialect-NEXT:   %13 = llvm.add %11, %0  : i64
!LLVMIRDialect-NEXT:   %14 = llvm.mul %9, %3  : i64
!LLVMIRDialect-NEXT:   %15 = llvm.mul %12, %3  : i64
!LLVMIRDialect-NEXT:   %16 = llvm.mul %13, %3  : i64
!LLVMIRDialect-NEXT:   %17 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %18 = llvm.insertvalue %arg1, %17[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %19 = llvm.insertvalue %14, %18[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %20 = llvm.insertvalue %15, %19[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %21 = llvm.insertvalue %16, %20[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   llvm.return %21 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep3(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:   %4 = llvm.load %arg1 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %5 = llvm.sext %4 : i32 to i64
!LLVMIRDialect-NEXT:   %6 = llvm.add %4, %2  : i32
!LLVMIRDialect-NEXT:   %7 = llvm.sext %6 : i32 to i64
!LLVMIRDialect-NEXT:   %8 = llvm.sub %7, %5  : i64
!LLVMIRDialect-NEXT:   %9 = llvm.add %8, %3  : i64
!LLVMIRDialect-NEXT:   %10 = llvm.icmp "sgt" %9, %1 : i64
!LLVMIRDialect-NEXT:   %11 = llvm.select %10, %9, %1 : i1, i64
!LLVMIRDialect-NEXT:   %12 = llvm.sub %5, %3  : i64
!LLVMIRDialect-NEXT:   %13 = llvm.mul %11, %0  : i64
!LLVMIRDialect-NEXT:   %14 = llvm.mul %12, %0  : i64
!LLVMIRDialect-NEXT:   %15 = llvm.mul %7, %0  : i64
!LLVMIRDialect-NEXT:   %16 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %17 = llvm.insertvalue %arg0, %16[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %18 = llvm.insertvalue %13, %17[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %19 = llvm.insertvalue %14, %18[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %20 = llvm.insertvalue %15, %19[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   llvm.return %20 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

!LLVMIRDialect-LABEL: llvm.func @compute.dep4(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect:   %0 = llvm.mlir.constant(-3 : i64) : i64
!LLVMIRDialect-NEXT:   %1 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT:   %2 = llvm.mlir.constant(6 : i32) : i32
!LLVMIRDialect-NEXT:   %3 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT:   %4 = llvm.load %arg0 : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:   %5 = llvm.add %4, %2  : i32
!LLVMIRDialect-NEXT:   %6 = llvm.sext %5 : i32 to i64
!LLVMIRDialect-NEXT:   %7 = llvm.add %6, %0  : i64
!LLVMIRDialect-NEXT:   %8 = llvm.icmp "sgt" %7, %1 : i64
!LLVMIRDialect-NEXT:   %9 = llvm.select %8, %7, %1 : i1, i64
!LLVMIRDialect-NEXT:   %10 = llvm.mul %9, %3  : i64
!LLVMIRDialect-NEXT:   %11 = llvm.mul %7, %3  : i64
!LLVMIRDialect-NEXT:   %12 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT:   %13 = llvm.insertvalue %arg1, %12[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %14 = llvm.insertvalue %10, %13[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %15 = llvm.insertvalue %1, %14[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   %16 = llvm.insertvalue %11, %15[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)> 
!LLVMIRDialect-NEXT:   llvm.return %16 : !llvm.struct<(ptr<i32>, i64, i64, i64)>
