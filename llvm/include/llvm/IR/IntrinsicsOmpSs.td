//===- IntrinsicsOmpSs.td - Defines OmpSs intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

