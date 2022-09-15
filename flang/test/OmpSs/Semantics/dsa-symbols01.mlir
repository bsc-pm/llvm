func @_QPs(%arg0: !fir.ref<!fir.array<10xi32>>, %arg1: !fir.ref<!fir.array<?xi32>>) {
  %c1 = constant 1 : index
  %c4_i32 = constant 4 : i32
  oss.task shared(%arg0 : !fir.ref<!fir.array<10xi32>>, %arg1 : !fir.ref<!fir.array<?xi32>>) {
    %0 = fir.coordinate_of %arg1, %c1 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %c4_i32 to %0 : !fir.ref<i32>
    oss.terminator
  }
  return
}
fir.global internal @_QFsECp constant : i32 {
  %c4_i32 = constant 4 : i32
  fir.has_value %c4_i32 : i32
}

