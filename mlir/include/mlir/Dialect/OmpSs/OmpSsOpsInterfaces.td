//===-- OmpSsOpsInterfaces.td - OmpSs op interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the OmpSs Dialect interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef OmpSs_OPS_INTERFACES
#define OmpSs_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def OutlineableOmpSsOpInterface : OpInterface<"OutlineableOmpSsOpInterface"> {
  let description = [{
    OmpSs operations whose region will be outlined will implement this
    interface.
  }];

  let cppNamespace = "::mlir::oss";

  let methods = [
    InterfaceMethod<"Get alloca block", "::mlir::Block*", "getAllocaBlock",
      (ins), [{
      return &$_op.getRegion().front();
      }]>,
  ];
}

#endif // OmpSs_OPS_INTERFACES
