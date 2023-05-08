//===- IntrinsicsOmpSs.td - Defines OmpSs intrinsics ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the OmpSs-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "directive" in {  // All intrinsics start with "llvm.directive.".

def int_directive_region_entry : Intrinsic<[llvm_token_ty], [], []>;
def int_directive_region_exit : Intrinsic<[], [llvm_token_ty], []>;
def int_directive_marker : Intrinsic<[llvm_i1_ty], [], []>;

}

