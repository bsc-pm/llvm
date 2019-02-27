//===-- OmpSs.h - OmpSs Transformations -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the OmpSs transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_OMPSS_H
#define LLVM_TRANSFORMS_OMPSS_H

#include <functional>

namespace llvm {

class BasicBlockPass;
class Function;
class FunctionPass;
class ModulePass;
class Pass;
class GetElementPtrInst;
class PassInfo;
class TargetLowering;
class TargetMachine;

//===----------------------------------------------------------------------===//
//
// OmpSsPass - Modifies IR in order to translate intrinsics to legal OmpSs-2 code
ModulePass *createOmpSsPass();

} // End llvm namespace

#endif
