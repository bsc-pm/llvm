//===- OmpSsToLLVM.h - Utils to convert from the OmpSs dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OMPSSTOLLVM_OMPSSTOLLVM_H_
#define MLIR_CONVERSION_OMPSSTOLLVM_OMPSSTOLLVM_H_

#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

#define GEN_PASS_DECL_CONVERTOOMPSSTOLLVM
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from OmpSs-2 to LLVM.
void populateOmpSsToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                           OwningRewritePatternList &patterns);

/// Configure conversion target with OmpSs-2 legal operations.
void configureOmpSsToLLVMConversionLegality(ConversionTarget &target,
                                             LLVMTypeConverter &typeConverter);

/// Create a pass to convert OmpSs operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertOmpSsToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OMPSSTOLLVM_OMPSSTOLLVM_H_
