//===- OmpSsDialect.h - MLIR Dialect for OmpSs ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the OmpSs dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OMPSS_OMPSSDIALECT_H_
#define MLIR_DIALECT_OMPSS_OMPSSDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/OmpSs/OmpSsOpsDialect.h.inc"
#include "mlir/Dialect/OmpSs/OmpSsOpsEnums.h.inc"
#include "mlir/Dialect/OmpSs/OmpSsOpsInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OmpSs/OmpSsOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OmpSs/OmpSsOps.h.inc"

#endif // MLIR_DIALECT_OMPSS_OMPSSDIALECT_H_
