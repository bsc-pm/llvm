//===- OmpSsToLLVM.cpp - conversion from OmpSs to LLVM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OmpSsToLLVM/ConvertOmpSsToLLVM.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/OmpSs/OmpSsDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTOMPSSTOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A pattern that converts the region arguments in a single-region OmpSs-2
/// operation to the LLVM dialect. The body of the region is not modified and is
/// expected to either be processed by the conversion infrastructure or already
/// contain ops compatible with LLVM dialect types.
template <typename OpType>
struct RegionOpConversion : public ConvertOpToLLVMPattern<OpType> {
  using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpType curOp, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<OpType>(
        curOp.getLoc(), TypeRange(), adaptor.getOperands(), curOp->getAttrs());
    rewriter.inlineRegionBefore(curOp.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *this->getTypeConverter())))
      return failure();

    rewriter.eraseOp(curOp);
    return success();
  }
};

template <typename T>
struct RegionLessOpWithVarOperandsConversion
    : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(T curOp, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    SmallVector<Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<T>(curOp, resTypes, adaptor.getOperands(),
                                   curOp->getAttrs());
    return success();
  }
};

} // namespace

void mlir::populateOmpSsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<RegionOpConversion<oss::TaskOp>>(converter);
  patterns.insert<RegionOpConversion<oss::TaskForOp>>(converter);
  patterns.insert<RegionOpConversion<oss::TaskloopOp>>(converter);
  patterns.insert<RegionOpConversion<oss::TaskloopForOp>>(converter);

  patterns.insert<RegionLessOpWithVarOperandsConversion<oss::DepOp>>(converter);
  patterns.insert<RegionLessOpWithVarOperandsConversion<oss::CopyOp>>(converter);
  patterns.insert<RegionLessOpWithVarOperandsConversion<oss::VlaDimOp>>(converter);
  patterns.insert<RegionLessOpWithVarOperandsConversion<oss::TaskwaitOp>>(converter);
  patterns.insert<RegionLessOpWithVarOperandsConversion<oss::ReleaseOp>>(converter);
}

void mlir::configureOmpSsToLLVMConversionLegality(
    ConversionTarget &target, LLVMTypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<
    oss::TaskOp, oss::TaskForOp, oss::TaskloopOp,
    oss::TaskloopForOp>(
      [&](Operation *op) {
        return typeConverter.isLegal(&op->getRegion(0)) &&
               typeConverter.isLegal(op->getOperandTypes());
      });
  target.addDynamicallyLegalOp<
    oss::DepOp, oss::CopyOp, oss::VlaDimOp>(
      [&](Operation *op) {
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
      });
  target.addDynamicallyLegalOp<
    oss::TaskwaitOp, oss::ReleaseOp>(
      [&](Operation *op) {
        return typeConverter.isLegal(op->getOperandTypes());
      });
  target.addLegalOp<oss::TerminatorOp>();
}

namespace {
struct ConvertOmpSsToLLVMPass
    : public impl::ConvertOmpSsToLLVMBase<ConvertOmpSsToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOmpSsToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to OmpSs-2 operations with LLVM IR dialect
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  mlir::arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateOmpSsToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  configureOmpSsToLLVMConversionLegality(target, converter);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOmpSsToLLVMPass() {
  return std::make_unique<ConvertOmpSsToLLVMPass>();
}
