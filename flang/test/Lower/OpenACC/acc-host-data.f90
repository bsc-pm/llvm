! This test checks lowering of OpenACC host_data directive.

! RUN: bbc -fopenacc -emit-fir -hlfir=false %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

subroutine acc_host_data()
  real, dimension(10) :: a
  logical :: ifCondition = .TRUE.

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "a", uniq_name = "_QFacc_host_dataEa"}
! HLFIR: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[IFCOND:.*]] = fir.address_of(@_QFacc_host_dataEifcondition) : !fir.ref<!fir.logical<4>>
! HLFIR: %[[DECLIFCOND:.*]]:2 = hlfir.declare %[[IFCOND]]

  !$acc host_data use_device(a)
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! FIR: %[[DA:.*]] = acc.use_device varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: acc.host_data dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>)

  !$acc host_data use_device(a) if_present
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! FIR: %[[DA:.*]] = acc.use_device varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: acc.host_data dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK: } attributes {ifPresent}

  !$acc host_data use_device(a) if(ifCondition)
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! FIR: %[[DA:.*]] = acc.use_device varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR: %[[LOAD_IFCOND:.*]] = fir.load %[[IFCOND]] : !fir.ref<!fir.logical<4>>
! HLFIR: %[[LOAD_IFCOND:.*]] = fir.load %[[DECLIFCOND]]#0 : !fir.ref<!fir.logical<4>>
! CHECK: %[[IFCOND_I1:.*]] = fir.convert %[[LOAD_IFCOND]] : (!fir.logical<4>) -> i1
! CHECK: acc.host_data if(%[[IFCOND_I1]]) dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>)

  !$acc host_data use_device(a) if(.true.)
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! FIR: %[[DA:.*]] = acc.use_device varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: acc.host_data dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>)

  !$acc host_data use_device(a) if(.false.)
    a = 1.0
  !$acc end host_data

! CHECK-NOT: acc.host_data
! FIR: fir.do_loop
! HLFIR: hlfir.assign %{{.*}} to %[[DECLA]]#0

end subroutine
