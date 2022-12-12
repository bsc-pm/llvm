! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

! - [x] assumed-shape
! - [x] deferred-shape

! NOTE: assumed-shape do not have init/copy/deinit
! functions

PROGRAM P
  IMPLICIT NONE
  INTEGER :: ARRAY(10)
  INTEGER, ALLOCATABLE :: GLOB_ALLOC(:)
  INTEGER, POINTER :: GLOB_PTR(:)

  !$OSS TASK SHARED(GLOB_ALLOC, GLOB_PTR)
  GLOB_ALLOC(7) = 7
  GLOB_PTR(7) = 7
  !$OSS END TASK

  !$OSS TASK PRIVATE(GLOB_ALLOC, GLOB_PTR)
  GLOB_ALLOC(7) = 7
  GLOB_PTR(7) = 7
  !$OSS END TASK

  !$OSS TASK FIRSTPRIVATE(GLOB_ALLOC, GLOB_PTR)
  GLOB_ALLOC(7) = 7
  GLOB_PTR(7) = 7
  !$OSS END TASK

  CALL FOO(GLOB_ALLOC, GLOB_PTR, ARRAY)

CONTAINS

SUBROUTINE FOO(DUMMY_ALLOC, DUMMY_PTR, DUMMY_ARRAY)
  IMPLICIT NONE
  INTEGER, ALLOCATABLE :: DUMMY_ALLOC(:)
  INTEGER, POINTER :: DUMMY_PTR(:)
  INTEGER :: DUMMY_ARRAY(:)
  INTEGER, ALLOCATABLE :: LOCAL_ALLOC(:)
  INTEGER, POINTER :: LOCAL_PTR(:)

  !$OSS TASK SHARED(DUMMY_ALLOC, DUMMY_PTR, DUMMY_ARRAY)
  DUMMY_ALLOC(7) = 7
  DUMMY_PTR(7) = 7
  DUMMY_ARRAY(7) = 7
  !$OSS END TASK

  !$OSS TASK PRIVATE(DUMMY_ALLOC, DUMMY_PTR, DUMMY_ARRAY)
  DUMMY_ALLOC(7) = 7
  DUMMY_PTR(7) = 7
  DUMMY_ARRAY(7) = 7
  !$OSS END TASK

  !$OSS TASK FIRSTPRIVATE(DUMMY_ALLOC, DUMMY_PTR, DUMMY_ARRAY)
  DUMMY_ALLOC(7) = 7
  DUMMY_PTR(7) = 7
  DUMMY_ARRAY(7) = 7
  !$OSS END TASK

  !$OSS TASK SHARED(LOCAL_ALLOC, LOCAL_PTR)
  LOCAL_ALLOC(7) = 7
  LOCAL_PTR(7) = 7
  !$OSS END TASK

  !$OSS TASK PRIVATE(LOCAL_ALLOC, LOCAL_PTR)
  LOCAL_ALLOC(7) = 7
  LOCAL_PTR(7) = 7
  !$OSS END TASK

  !$OSS TASK FIRSTPRIVATE(LOCAL_ALLOC, LOCAL_PTR)
  LOCAL_ALLOC(7) = 7
  LOCAL_PTR(7) = 7
  !$OSS END TASK

END SUBROUTINE
END PROGRAM

! These are checks for described variables pack and unpack
!FIRDialect-LABEL: func.func @_QFPfoo
!FIRDialect: %17 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT: %18 = fir.load %3 : !fir.ref<index>
!FIRDialect-NEXT: %19 = fir.load %1 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT: %20 = fir.shape_shift %17, %18 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT: %21 = fir.embox %19(%20) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT: fir.store %21 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT: oss.task shared(%0, %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
!FIRDialect-NEXT:   %40 = fir.alloca index {bindc_name = "extent", pinned}
!FIRDialect-NEXT:   %41 = fir.alloca index {bindc_name = "lb", pinned}
!FIRDialect-NEXT:   %42 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr", pinned}
!FIRDialect-NEXT:   %43 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0_0 = arith.constant 0 : index
!FIRDialect-NEXT:   %44:3 = fir.box_dims %43, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %45 = fir.box_addr %43 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %45 to %42 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %44#1 to %40 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %44#0 to %41 : !fir.ref<index>

!FIRDialect:  %22 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT:  %23 = fir.load %3 : !fir.ref<index>
!FIRDialect-NEXT:  %24 = fir.load %1 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:  %25 = fir.shape_shift %22, %23 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:  %26 = fir.embox %24(%25) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:  fir.store %26 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect:  oss.task private(%5 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) firstprivate(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) copy(%27 : i32) init(%29 : i32) deinit(%28, %30 : i32, i32) {
!FIRDialect-NEXT:    %40 = fir.alloca index {bindc_name = "extent", pinned}
!FIRDialect-NEXT:    %41 = fir.alloca index {bindc_name = "lb", pinned}
!FIRDialect-NEXT:    %42 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr", pinned}
!FIRDialect-NEXT:    %43 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:    %c0_0 = arith.constant 0 : index
!FIRDialect-NEXT:    %44:3 = fir.box_dims %43, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:    %45 = fir.box_addr %43 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:    fir.store %45 to %42 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:    fir.store %44#1 to %40 : !fir.ref<index>
!FIRDialect-NEXT:    fir.store %44#0 to %41 : !fir.ref<index>

!FIRDialect:  %31 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT:  %32 = fir.load %3 : !fir.ref<index>
!FIRDialect-NEXT:  %33 = fir.load %1 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:  %34 = fir.shape_shift %31, %32 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:  %35 = fir.embox %33(%34) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:  fir.store %35 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect:  oss.task firstprivate(%0, %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) copy(%36, %38 : i32, i32) deinit(%37, %39 : i32, i32) {
!FIRDialect-NEXT:    %40 = fir.alloca index {bindc_name = "extent", pinned}
!FIRDialect-NEXT:    %41 = fir.alloca index {bindc_name = "lb", pinned}
!FIRDialect-NEXT:    %42 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr", pinned}
!FIRDialect-NEXT:    %43 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:    %c0_0 = arith.constant 0 : index
!FIRDialect-NEXT:    %44:3 = fir.box_dims %43, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:    %45 = fir.box_addr %43 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:    fir.store %45 to %42 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:    fir.store %44#1 to %40 : !fir.ref<index>
!FIRDialect-NEXT:    fir.store %44#0 to %41 : !fir.ref<index>

!FIRDialect-LABEL: func.func @compute.copy0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %1 = fir.box_addr %0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %2 = fir.convert %1 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %3 = arith.cmpi ne, %2, %c0_i64 : i64
!FIRDialect-NEXT:   fir.if %3 {
!FIRDialect-NEXT:     %4 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %c0 = arith.constant 0 : index
!FIRDialect-NEXT:     %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:     %6 = fir.box_addr %4 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %c1 = arith.constant 1 : index
!FIRDialect-NEXT:     %7 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:     %8 = fir.array_load %6(%7) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:     %9 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %10 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %11 = fir.convert %10 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:     %c0_i64_0 = arith.constant 0 : i64
!FIRDialect-NEXT:     %12 = arith.cmpi ne, %11, %c0_i64_0 : i64
!FIRDialect-NEXT:     %13:2 = fir.if %12 -> (i1, !fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:       %false = arith.constant false
!FIRDialect-NEXT:       %c0_1 = arith.constant 0 : index
!FIRDialect-NEXT:       %14:3 = fir.box_dims %9, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:       %15 = arith.cmpi ne, %14#1, %5#1 : index
!FIRDialect-NEXT:       %16 = arith.select %15, %15, %false : i1
!FIRDialect-NEXT:       %17 = fir.if %16 -> (!fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:         %18 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:         fir.result %18 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       } else {
!FIRDialect-NEXT:         fir.result %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.result %16, %17 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     } else {
!FIRDialect-NEXT:       %true = arith.constant true
!FIRDialect-NEXT:       %14 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:       fir.result %true, %14 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:     fir.if %13#0 {
!FIRDialect-NEXT:       %14 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:       fir.if %12 {
!FIRDialect-NEXT:         fir.freemem %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %15 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:       %16 = fir.embox %13#1(%15) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:       fir.store %16 to %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:   }

!FIRDialect-LABEL: func.func @compute.deinit0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %false = arith.constant false
!FIRDialect-NEXT:   %0 = fir.absent !fir.box<none>
!FIRDialect-NEXT:   %1 = fir.zero_bits !fir.ref<none>
!FIRDialect-NEXT:   %c0_i32 = arith.constant 0 : i32
!FIRDialect-NEXT:   %2 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.freemem %3 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %4 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %5 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %6 = fir.embox %4(%5) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %6 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRBuilder-LABEL: func.func @compute.init0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRBuilder:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRBuilder-NEXT:   %c0 = arith.constant 0 : index
!FIRBuilder-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRBuilder-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRBuilder-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect: func.func @compute.deinit1(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect-NEXT:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy1(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %1 = fir.box_addr %0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %2 = fir.convert %1 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %3 = arith.cmpi ne, %2, %c0_i64 : i64
!FIRDialect-NEXT:   fir.if %3 {
!FIRDialect-NEXT:     %4 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %c0 = arith.constant 0 : index
!FIRDialect-NEXT:     %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:     %6 = fir.box_addr %4 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %c1 = arith.constant 1 : index
!FIRDialect-NEXT:     %7 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:     %8 = fir.array_load %6(%7) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:     %9 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %10 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %11 = fir.convert %10 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:     %c0_i64_0 = arith.constant 0 : i64
!FIRDialect-NEXT:     %12 = arith.cmpi ne, %11, %c0_i64_0 : i64
!FIRDialect-NEXT:     %13:2 = fir.if %12 -> (i1, !fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:       %false = arith.constant false
!FIRDialect-NEXT:       %c0_1 = arith.constant 0 : index
!FIRDialect-NEXT:       %14:3 = fir.box_dims %9, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:       %15 = arith.cmpi ne, %14#1, %5#1 : index
!FIRDialect-NEXT:       %16 = arith.select %15, %15, %false : i1
!FIRDialect-NEXT:       %17 = fir.if %16 -> (!fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:         %18 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:         %19 = fir.shape %5#1 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:         %20 = fir.array_load %18(%19) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:         %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:         %21 = arith.subi %5#1, %c1_2 : index
!FIRDialect-NEXT:         %22 = fir.do_loop %arg3 = %c0_3 to %21 step %c1_2 unordered iter_args(%arg4 = %20) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:           %24 = fir.array_fetch %8, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:           %25 = fir.array_update %arg4, %24, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:           %26 = fir.undefined index
!FIRDialect-NEXT:           fir.result %25 : !fir.array<?xi32>
!FIRDialect-NEXT:         }
!FIRDialect-NEXT:         %23 = fir.undefined index
!FIRDialect-NEXT:         fir.array_merge_store %20, %22 to %18 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:         fir.result %18 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       } else {
!FIRDialect-NEXT:         %18 = fir.shape %5#1 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:         %19 = fir.array_load %10(%18) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:         %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:         %20 = arith.subi %5#1, %c1_2 : index
!FIRDialect-NEXT:         %21 = fir.do_loop %arg3 = %c0_3 to %20 step %c1_2 unordered iter_args(%arg4 = %19) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:           %23 = fir.array_fetch %8, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:           %24 = fir.array_update %arg4, %23, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:           %25 = fir.undefined index
!FIRDialect-NEXT:           fir.result %24 : !fir.array<?xi32>
!FIRDialect-NEXT:         }
!FIRDialect-NEXT:         %22 = fir.undefined index
!FIRDialect-NEXT:         fir.array_merge_store %19, %21 to %10 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:         fir.result %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.result %16, %17 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     } else {
!FIRDialect-NEXT:       %true = arith.constant true
!FIRDialect-NEXT:       %14 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:       %15 = fir.shape %5#1 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:       %16 = fir.array_load %14(%15) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:       %c1_1 = arith.constant 1 : index
!FIRDialect-NEXT:       %c0_2 = arith.constant 0 : index
!FIRDialect-NEXT:       %17 = arith.subi %5#1, %c1_1 : index
!FIRDialect-NEXT:       %18 = fir.do_loop %arg3 = %c0_2 to %17 step %c1_1 unordered iter_args(%arg4 = %16) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:         %20 = fir.array_fetch %8, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:         %21 = fir.array_update %arg4, %20, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %22 = fir.undefined index
!FIRDialect-NEXT:         fir.result %21 : !fir.array<?xi32>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %19 = fir.undefined index
!FIRDialect-NEXT:       fir.array_merge_store %16, %18 to %14 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       fir.result %true, %14 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:     fir.if %13#0 {
!FIRDialect-NEXT:       %14 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:       fir.if %12 {
!FIRDialect-NEXT:         fir.freemem %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %15 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:       %16 = fir.embox %13#1(%15) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:       fir.store %16 to %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:   }

!FIRDialect-LABEL: func.func @compute.deinit2(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %false = arith.constant false
!FIRDialect-NEXT:   %0 = fir.absent !fir.box<none>
!FIRDialect-NEXT:   %1 = fir.zero_bits !fir.ref<none>
!FIRDialect-NEXT:   %c0_i32 = arith.constant 0 : i32
!FIRDialect-NEXT:   %2 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.freemem %3 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %4 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %5 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %6 = fir.embox %4(%5) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %6 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy2(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %2 = fir.shift %1#0 : (index) -> !fir.shift<1>
!FIRDialect-NEXT:   %3 = fir.rebox %0(%2) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %3 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit3(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy3(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %1 = fir.box_addr %0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %2 = fir.convert %1 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %3 = arith.cmpi ne, %2, %c0_i64 : i64
!FIRDialect-NEXT:   fir.if %3 {
!FIRDialect-NEXT:     %4 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %c0 = arith.constant 0 : index
!FIRDialect-NEXT:     %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:     %6 = fir.box_addr %4 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %c1 = arith.constant 1 : index
!FIRDialect-NEXT:     %7 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:     %8 = fir.array_load %6(%7) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:     %9 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %10 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %11 = fir.convert %10 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:     %c0_i64_0 = arith.constant 0 : i64
!FIRDialect-NEXT:     %12 = arith.cmpi ne, %11, %c0_i64_0 : i64
!FIRDialect-NEXT:     %13:2 = fir.if %12 -> (i1, !fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:       %false = arith.constant false
!FIRDialect-NEXT:       %c0_1 = arith.constant 0 : index
!FIRDialect-NEXT:       %14:3 = fir.box_dims %9, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:       %15 = arith.cmpi ne, %14#1, %5#1 : index
!FIRDialect-NEXT:       %16 = arith.select %15, %15, %false : i1
!FIRDialect-NEXT:       %17 = fir.if %16 -> (!fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:         %18 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:         fir.result %18 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       } else {
!FIRDialect-NEXT:         fir.result %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.result %16, %17 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     } else {
!FIRDialect-NEXT:       %true = arith.constant true
!FIRDialect-NEXT:       %14 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:       fir.result %true, %14 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:     fir.if %13#0 {
!FIRDialect-NEXT:       %14 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:       fir.if %12 {
!FIRDialect-NEXT:         fir.freemem %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %15 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:       %16 = fir.embox %13#1(%15) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:       fir.store %16 to %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:   }

!FIRDialect-LABEL: func.func @compute.deinit4(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %false = arith.constant false
!FIRDialect-NEXT:   %0 = fir.absent !fir.box<none>
!FIRDialect-NEXT:   %1 = fir.zero_bits !fir.ref<none>
!FIRDialect-NEXT:   %c0_i32 = arith.constant 0 : i32
!FIRDialect-NEXT:   %2 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.freemem %3 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %4 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %5 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %6 = fir.embox %4(%5) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %6 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.init1(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit5(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy4(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %1 = fir.box_addr %0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %2 = fir.convert %1 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %3 = arith.cmpi ne, %2, %c0_i64 : i64
!FIRDialect-NEXT:   fir.if %3 {
!FIRDialect-NEXT:     %4 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %c0 = arith.constant 0 : index
!FIRDialect-NEXT:     %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:     %6 = fir.box_addr %4 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %c1 = arith.constant 1 : index
!FIRDialect-NEXT:     %7 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:     %8 = fir.array_load %6(%7) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:     %9 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     %10 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     %11 = fir.convert %10 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:     %c0_i64_0 = arith.constant 0 : i64
!FIRDialect-NEXT:     %12 = arith.cmpi ne, %11, %c0_i64_0 : i64
!FIRDialect-NEXT:     %13:2 = fir.if %12 -> (i1, !fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:       %false = arith.constant false
!FIRDialect-NEXT:       %c0_1 = arith.constant 0 : index
!FIRDialect-NEXT:       %14:3 = fir.box_dims %9, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:       %15 = arith.cmpi ne, %14#1, %5#1 : index
!FIRDialect-NEXT:       %16 = arith.select %15, %15, %false : i1
!FIRDialect-NEXT:       %17 = fir.if %16 -> (!fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:         %18 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:         %19 = fir.shape %5#1 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:         %20 = fir.array_load %18(%19) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:         %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:         %21 = arith.subi %5#1, %c1_2 : index
!FIRDialect-NEXT:         %22 = fir.do_loop %arg3 = %c0_3 to %21 step %c1_2 unordered iter_args(%arg4 = %20) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:           %24 = fir.array_fetch %8, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:           %25 = fir.array_update %arg4, %24, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:           %26 = fir.undefined index
!FIRDialect-NEXT:           fir.result %25 : !fir.array<?xi32>
!FIRDialect-NEXT:         }
!FIRDialect-NEXT:         %23 = fir.undefined index
!FIRDialect-NEXT:         fir.array_merge_store %20, %22 to %18 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:         fir.result %18 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       } else {
!FIRDialect-NEXT:         %18 = fir.shape %5#1 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:         %19 = fir.array_load %10(%18) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:         %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:         %20 = arith.subi %5#1, %c1_2 : index
!FIRDialect-NEXT:         %21 = fir.do_loop %arg3 = %c0_3 to %20 step %c1_2 unordered iter_args(%arg4 = %19) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:           %23 = fir.array_fetch %8, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:           %24 = fir.array_update %arg4, %23, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:           %25 = fir.undefined index
!FIRDialect-NEXT:           fir.result %24 : !fir.array<?xi32>
!FIRDialect-NEXT:         }
!FIRDialect-NEXT:         %22 = fir.undefined index
!FIRDialect-NEXT:         fir.array_merge_store %19, %21 to %10 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:         fir.result %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.result %16, %17 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     } else {
!FIRDialect-NEXT:       %true = arith.constant true
!FIRDialect-NEXT:       %14 = fir.allocmem !fir.array<?xi32>, %5#1 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:       %15 = fir.shape %5#1 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:       %16 = fir.array_load %14(%15) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:       %c1_1 = arith.constant 1 : index
!FIRDialect-NEXT:       %c0_2 = arith.constant 0 : index
!FIRDialect-NEXT:       %17 = arith.subi %5#1, %c1_1 : index
!FIRDialect-NEXT:       %18 = fir.do_loop %arg3 = %c0_2 to %17 step %c1_1 unordered iter_args(%arg4 = %16) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:         %20 = fir.array_fetch %8, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:         %21 = fir.array_update %arg4, %20, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %22 = fir.undefined index
!FIRDialect-NEXT:         fir.result %21 : !fir.array<?xi32>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %19 = fir.undefined index
!FIRDialect-NEXT:       fir.array_merge_store %16, %18 to %14 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       fir.result %true, %14 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:     fir.if %13#0 {
!FIRDialect-NEXT:       %14 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:       fir.if %12 {
!FIRDialect-NEXT:         fir.freemem %10 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %15 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:       %16 = fir.embox %13#1(%15) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:       fir.store %16 to %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:   }

!FIRDialect-LABEL: func.func @compute.deinit6(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %false = arith.constant false
!FIRDialect-NEXT:   %0 = fir.absent !fir.box<none>
!FIRDialect-NEXT:   %1 = fir.zero_bits !fir.ref<none>
!FIRDialect-NEXT:   %c0_i32 = arith.constant 0 : i32
!FIRDialect-NEXT:   %2 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.freemem %3 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %4 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %5 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %6 = fir.embox %4(%5) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %6 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy5(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %2 = fir.shift %1#0 : (index) -> !fir.shift<1>
!FIRDialect-NEXT:   %3 = fir.rebox %0(%2) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %3 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit7(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy6(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.alloca index {bindc_name = "extent.dst"}
!FIRDialect-NEXT:   %1 = fir.alloca index {bindc_name = "extent.src"}
!FIRDialect-NEXT:   %2 = fir.alloca index {bindc_name = "lb.dst"}
!FIRDialect-NEXT:   %3 = fir.alloca index {bindc_name = "lb.src"}
!FIRDialect-NEXT:   %4 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr.dst"}
!FIRDialect-NEXT:   %5 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr.src"}
!FIRDialect-NEXT:   %6 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %7:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %8 = fir.box_addr %6 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %8 to %5 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %7#1 to %1 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %7#0 to %3 : !fir.ref<index>
!FIRDialect-NEXT:   %9 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0_0 = arith.constant 0 : index
!FIRDialect-NEXT:   %10:3 = fir.box_dims %9, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %11 to %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %10#1 to %0 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %10#0 to %2 : !fir.ref<index>
!FIRDialect-NEXT:   %12 = fir.load %5 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %13 = fir.convert %12 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %14 = arith.cmpi ne, %13, %c0_i64 : i64
!FIRDialect-NEXT:   fir.if %14 {
!FIRDialect-NEXT:     %20 = fir.load %3 : !fir.ref<index>
!FIRDialect-NEXT:     %21 = fir.load %1 : !fir.ref<index>
!FIRDialect-NEXT:     %22 = fir.load %5 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:     %c1 = arith.constant 1 : index
!FIRDialect-NEXT:     %23 = fir.shape_shift %20, %21 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:     %24 = fir.array_load %22(%23) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:     %25 = fir.load %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:     %26 = fir.convert %25 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:     %c0_i64_1 = arith.constant 0 : i64
!FIRDialect-NEXT:     %27 = arith.cmpi ne, %26, %c0_i64_1 : i64
!FIRDialect-NEXT:     %28:2 = fir.if %27 -> (i1, !fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:       %false = arith.constant false
!FIRDialect-NEXT:       %29 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT:       %30 = fir.load %0 : !fir.ref<index>
!FIRDialect-NEXT:       %31 = arith.cmpi ne, %30, %21 : index
!FIRDialect-NEXT:       %32 = arith.select %31, %31, %false : i1
!FIRDialect-NEXT:       %33 = fir.if %32 -> (!fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:         %34 = fir.allocmem !fir.array<?xi32>, %21 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:         fir.result %34 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       } else {
!FIRDialect-NEXT:         fir.result %25 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.result %32, %33 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     } else {
!FIRDialect-NEXT:       %true = arith.constant true
!FIRDialect-NEXT:       %29 = fir.allocmem !fir.array<?xi32>, %21 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:       fir.result %true, %29 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:     fir.if %28#0 {
!FIRDialect-NEXT:       fir.if %27 {
!FIRDialect-NEXT:         fir.freemem %25 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.store %28#1 to %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:       fir.store %21 to %0 : !fir.ref<index>
!FIRDialect-NEXT:       fir.store %20 to %2 : !fir.ref<index>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:   }
!FIRDialect-NEXT:   %15 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT:   %16 = fir.load %0 : !fir.ref<index>
!FIRDialect-NEXT:   %17 = fir.load %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %18 = fir.shape_shift %15, %16 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:   %19 = fir.embox %17(%18) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %19 to %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit8(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.alloca index {bindc_name = "extent.dst"}
!FIRDialect-NEXT:   %1 = fir.alloca index {bindc_name = "lb.dst"}
!FIRDialect-NEXT:   %2 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr.dst"}
!FIRDialect-NEXT:   %3 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %5 = fir.box_addr %3 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %5 to %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %4#1 to %0 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %4#0 to %1 : !fir.ref<index>
!FIRDialect-NEXT:   %false = arith.constant false
!FIRDialect-NEXT:   %6 = fir.absent !fir.box<none>
!FIRDialect-NEXT:   %7 = fir.zero_bits !fir.ref<none>
!FIRDialect-NEXT:   %c0_i32 = arith.constant 0 : i32
!FIRDialect-NEXT:   %8 = fir.load %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.freemem %8 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %9 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %9 to %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %10 = fir.load %1 : !fir.ref<index>
!FIRDialect-NEXT:   %11 = fir.load %0 : !fir.ref<index>
!FIRDialect-NEXT:   %12 = fir.load %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %13 = fir.shape_shift %10, %11 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:   %14 = fir.embox %12(%13) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %14 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.init2(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit9(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy7(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.alloca index {bindc_name = "extent.dst"}
!FIRDialect-NEXT:   %1 = fir.alloca index {bindc_name = "extent.src"}
!FIRDialect-NEXT:   %2 = fir.alloca index {bindc_name = "lb.dst"}
!FIRDialect-NEXT:   %3 = fir.alloca index {bindc_name = "lb.src"}
!FIRDialect-NEXT:   %4 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr.dst"}
!FIRDialect-NEXT:   %5 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr.src"}
!FIRDialect-NEXT:   %6 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %7:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %8 = fir.box_addr %6 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %8 to %5 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %7#1 to %1 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %7#0 to %3 : !fir.ref<index>
!FIRDialect-NEXT:   %9 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0_0 = arith.constant 0 : index
!FIRDialect-NEXT:   %10:3 = fir.box_dims %9, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %11 to %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %10#1 to %0 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %10#0 to %2 : !fir.ref<index>
!FIRDialect-NEXT:   %12 = fir.load %5 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %13 = fir.convert %12 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %14 = arith.cmpi ne, %13, %c0_i64 : i64
!FIRDialect-NEXT:   fir.if %14 {
!FIRDialect-NEXT:     %20 = fir.load %3 : !fir.ref<index>
!FIRDialect-NEXT:     %21 = fir.load %1 : !fir.ref<index>
!FIRDialect-NEXT:     %22 = fir.load %5 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:     %c1 = arith.constant 1 : index
!FIRDialect-NEXT:     %23 = fir.shape_shift %20, %21 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:     %24 = fir.array_load %22(%23) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:     %25 = fir.load %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:     %26 = fir.convert %25 : (!fir.heap<!fir.array<?xi32>>) -> i64
!FIRDialect-NEXT:     %c0_i64_1 = arith.constant 0 : i64
!FIRDialect-NEXT:     %27 = arith.cmpi ne, %26, %c0_i64_1 : i64
!FIRDialect-NEXT:     %28:2 = fir.if %27 -> (i1, !fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:       %false = arith.constant false
!FIRDialect-NEXT:       %29 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT:       %30 = fir.load %0 : !fir.ref<index>
!FIRDialect-NEXT:       %31 = arith.cmpi ne, %30, %21 : index
!FIRDialect-NEXT:       %32 = arith.select %31, %31, %false : i1
!FIRDialect-NEXT:       %33 = fir.if %32 -> (!fir.heap<!fir.array<?xi32>>) {
!FIRDialect-NEXT:         %34 = fir.allocmem !fir.array<?xi32>, %21 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:         %35 = fir.shape %21 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:         %36 = fir.array_load %34(%35) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:         %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:         %37 = arith.subi %21, %c1_2 : index
!FIRDialect-NEXT:         %38 = fir.do_loop %arg3 = %c0_3 to %37 step %c1_2 unordered iter_args(%arg4 = %36) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:           %40 = fir.array_fetch %24, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:           %41 = fir.array_update %arg4, %40, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:           %42 = fir.undefined index
!FIRDialect-NEXT:           fir.result %41 : !fir.array<?xi32>
!FIRDialect-NEXT:         }
!FIRDialect-NEXT:         %39 = fir.undefined index
!FIRDialect-NEXT:         fir.array_merge_store %36, %38 to %34 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:         fir.result %34 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       } else {
!FIRDialect-NEXT:         %34 = fir.shape %21 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:         %35 = fir.array_load %25(%34) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:         %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:         %36 = arith.subi %21, %c1_2 : index
!FIRDialect-NEXT:         %37 = fir.do_loop %arg3 = %c0_3 to %36 step %c1_2 unordered iter_args(%arg4 = %35) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:           %39 = fir.array_fetch %24, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:           %40 = fir.array_update %arg4, %39, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:           %41 = fir.undefined index
!FIRDialect-NEXT:           fir.result %40 : !fir.array<?xi32>
!FIRDialect-NEXT:         }
!FIRDialect-NEXT:         %38 = fir.undefined index
!FIRDialect-NEXT:         fir.array_merge_store %35, %37 to %25 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:         fir.result %25 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.result %32, %33 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     } else {
!FIRDialect-NEXT:       %true = arith.constant true
!FIRDialect-NEXT:       %29 = fir.allocmem !fir.array<?xi32>, %21 {uniq_name = ".auto.alloc"}
!FIRDialect-NEXT:       %30 = fir.shape %21 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:       %31 = fir.array_load %29(%30) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
!FIRDialect-NEXT:       %c1_2 = arith.constant 1 : index
!FIRDialect-NEXT:       %c0_3 = arith.constant 0 : index
!FIRDialect-NEXT:       %32 = arith.subi %21, %c1_2 : index
!FIRDialect-NEXT:       %33 = fir.do_loop %arg3 = %c0_3 to %32 step %c1_2 unordered iter_args(%arg4 = %31) -> (!fir.array<?xi32>) {
!FIRDialect-NEXT:         %35 = fir.array_fetch %24, %arg3 : (!fir.array<?xi32>, index) -> i32
!FIRDialect-NEXT:         %36 = fir.array_update %arg4, %35, %arg3 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
!FIRDialect-NEXT:         %37 = fir.undefined index
!FIRDialect-NEXT:         fir.result %36 : !fir.array<?xi32>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       %34 = fir.undefined index
!FIRDialect-NEXT:       fir.array_merge_store %31, %33 to %29 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       fir.result %true, %29 : i1, !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:     fir.if %28#0 {
!FIRDialect-NEXT:       fir.if %27 {
!FIRDialect-NEXT:         fir.freemem %25 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:       }
!FIRDialect-NEXT:       fir.store %28#1 to %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:       fir.store %21 to %0 : !fir.ref<index>
!FIRDialect-NEXT:       fir.store %20 to %2 : !fir.ref<index>
!FIRDialect-NEXT:     }
!FIRDialect-NEXT:   }
!FIRDialect-NEXT:   %15 = fir.load %2 : !fir.ref<index>
!FIRDialect-NEXT:   %16 = fir.load %0 : !fir.ref<index>
!FIRDialect-NEXT:   %17 = fir.load %4 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %18 = fir.shape_shift %15, %16 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:   %19 = fir.embox %17(%18) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %19 to %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit10(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.alloca index {bindc_name = "extent.dst"}
!FIRDialect-NEXT:   %1 = fir.alloca index {bindc_name = "lb.dst"}
!FIRDialect-NEXT:   %2 = fir.alloca !fir.heap<!fir.array<?xi32>> {bindc_name = "addr.dst"}
!FIRDialect-NEXT:   %3 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %5 = fir.box_addr %3 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %5 to %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %4#1 to %0 : !fir.ref<index>
!FIRDialect-NEXT:   fir.store %4#0 to %1 : !fir.ref<index>
!FIRDialect-NEXT:   %false = arith.constant false
!FIRDialect-NEXT:   %6 = fir.absent !fir.box<none>
!FIRDialect-NEXT:   %7 = fir.zero_bits !fir.ref<none>
!FIRDialect-NEXT:   %c0_i32 = arith.constant 0 : i32
!FIRDialect-NEXT:   %8 = fir.load %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.freemem %8 : !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   %9 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!FIRDialect-NEXT:   fir.store %9 to %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %10 = fir.load %1 : !fir.ref<index>
!FIRDialect-NEXT:   %11 = fir.load %0 : !fir.ref<index>
!FIRDialect-NEXT:   %12 = fir.load %2 : !fir.ref<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   %13 = fir.shape_shift %10, %11 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-NEXT:   %14 = fir.embox %12(%13) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %14 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.copy8(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg2: i64)
!FIRDialect:   %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %2 = fir.shift %1#0 : (index) -> !fir.shift<1>
!FIRDialect-NEXT:   %3 = fir.rebox %0(%2) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %3 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

!FIRDialect-LABEL: func.func @compute.deinit11(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: i64)
!FIRDialect:   %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %1 = fir.shape %c0 : (index) -> !fir.shape<1>
!FIRDialect-NEXT:   %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect-NEXT:   fir.store %2 to %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
