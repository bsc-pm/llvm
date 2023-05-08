//===-- OmpSs.cpp -- Open MP directive lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OmpSs.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/BoxAnalyzer.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/OmpSs/OmpSsDialect.h"
#include "llvm/Support/Debug.h"

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static mlir::Value addOperands(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymbolRef sym) {
  const auto variable = converter.getSymbolAddress(sym);
  if (variable) {
    return variable;
  } else {
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      converter.copySymbolBinding(details->symbol(), sym);
      return converter.getSymbolAddress(details->symbol());
    }
  }
  llvm_unreachable("addOperands");
}


static void genObjectList(const Fortran::parser::OSSObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          llvm::SmallVectorImpl<mlir::Value> &operands) {
  for (const auto &ossObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name = getDesignatorNameIfDataRef(designator)) {
                operands.push_back(addOperands(converter, *name->symbol));
              }
            },
            [&](const Fortran::parser::Name &name) {
              operands.push_back(addOperands(converter, *name.symbol));
            }},
        ossObject.u);
  }
}

static void genObjectList(const Fortran::parser::OSSObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          llvm::SetVector<mlir::Value> &operands) {
  for (const auto &ossObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name = getDesignatorNameIfDataRef(designator)) {
                operands.insert(addOperands(converter, *name->symbol));
              }
            },
            [&](const Fortran::parser::Name &name) {
              operands.insert(addOperands(converter, *name.symbol));
            }},
        ossObject.u);
  }
}

/// Create empty blocks for the current region.
/// These blocks replace blocks parented to an enclosing region.
static void createEmptyRegionBlocks(
    fir::FirOpBuilder &firOpBuilder,
    std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
  mlir::Region *region = &firOpBuilder.getRegion();
  for (Fortran::lower::pft::Evaluation &eval : evaluationList) {
    if (eval.block) {
      if (eval.block->empty()) {
        eval.block->erase();
        eval.block = firOpBuilder.createBlock(region);
      } else {
        [[maybe_unused]] mlir::Operation &terminatorOp = eval.block->back();
        assert((mlir::isa<mlir::oss::TerminatorOp>(terminatorOp)) &&
               "expected terminator op");
        // FIXME: Some subset of cases may need to insert a branch,
        // although this could be handled elsewhere.
        // if (?) {
        //   auto insertPt = firOpBuilder.saveInsertionPoint();
        //   firOpBuilder.setInsertionPointAfter(region->getParentOp());
        //   firOpBuilder.create<mlir::BranchOp>(
        //       terminatorOp.getLoc(), eval.block);
        //   firOpBuilder.restoreInsertionPoint(insertPt);
        // }
      }
    }
    if (!eval.isDirective() && eval.hasNestedEvaluations())
      createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());
  }
}

template <typename Op>
static void createBodyOfOp(Op &op, fir::FirOpBuilder &firOpBuilder,
                           mlir::Location &loc, Fortran::lower::pft::Evaluation &eval) {
  firOpBuilder.createBlock(&op.getRegion());

  auto &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);

  if (eval.lowerAsUnstructured())
    createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());
  // Ensure the block is well-formed.
  firOpBuilder.create<mlir::oss::TerminatorOp>(loc);
  // Reset the insertion point to the start of the first block.
  firOpBuilder.setInsertionPointToStart(&block);
}

template <typename Op>
static void createBodyOfOpWithPreface(Op &op, Fortran::lower::AbstractConverter &converter,
                           fir::FirOpBuilder &firOpBuilder,
                           mlir::Location &loc, Fortran::lower::pft::Evaluation &eval,
                           const Fortran::lower::ImplicitDSAs &implicitDSAs) {
  firOpBuilder.createBlock(&op.getRegion());

  auto &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);

  if (eval.lowerAsUnstructured())
    createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());

  // Ensure the block is well-formed.
  firOpBuilder.create<mlir::oss::TerminatorOp>(loc);
  // Reset the insertion point to the start of the first block.
  firOpBuilder.setInsertionPointToStart(&block);

  // Build new unpacked values inside the task and sync
  auto dupUnpackBoxes = [&converter, &firOpBuilder](const std::vector<Fortran::evaluate::SymbolRef> &list) {
    for (const auto &sym : list) {
      const auto variable = converter.getSymbolAddress(sym);
      if (variable) {
        fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym);
        if (auto *box = exv.getBoxOf<fir::MutableBoxValue>()) {
          if (box->isDescribedByVariables()) {
            auto loc = converter.genUnknownLocation();
            auto rank = box->rank();
            const auto &mutableProperties = box->getMutableProperties();

            fir::MutableProperties newMutableProperties;
            newMutableProperties.addr = firOpBuilder.createTemporary(loc, box->getMemTy(), "addr");

            // TODO: Idk if we have to check if the values exists
            for (decltype(rank) dim = 0; dim < rank; ++dim) {
              if (mutableProperties.lbounds[dim]) {
                newMutableProperties.lbounds.push_back(
                  firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "lb"));
              }
              if (mutableProperties.extents[dim]) {
                newMutableProperties.extents.push_back(
                  firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "extent"));
              }
            }
            auto newBox = fir::MutableBoxValue(
              fir::getBase(exv), box->nonDeferredLenParams(), newMutableProperties);

            // 2899b42f075fe8c016efb111b023ec6e8742642e dice que pushear un scope va a hacer
            // que no se encuentre el iterador del bucle al emitir la op de oss.taskloop
            converter.getLocalSymbols().addAllocatableOrPointer(sym, newBox);
            fir::factory::syncMutableBoxFromIRBox(firOpBuilder, loc, newBox);
          }
        }
      }
    }
  };
  // We do not want to dup shareds because they will be
  // directly captured. Both the box and the unpacked values
  dupUnpackBoxes(implicitDSAs.sharedList);
  dupUnpackBoxes(implicitDSAs.privateList);
  dupUnpackBoxes(implicitDSAs.firstprivateList);
}

namespace {
  class OSSDependInfoGathering {
  public:
    explicit OSSDependInfoGathering(
        Fortran::lower::AbstractConverter &converter,
        Fortran::semantics::SemanticsContext &context,
        Fortran::lower::StatementContext &stmtCtx)
      : converter(converter), context(context), stmtCtx(stmtCtx) {}

    template <typename A> void Walk(const A &x) { Fortran::parser::Walk(x, *this); }
    template <typename A> bool Pre(const A &) { return true; }
    template <typename A> void Post(const A &) {}
    bool Pre(const Fortran::parser::StructureComponent &x) {
      if (retOperands_.empty()) {
        // TODO: location
        auto loc = converter.genUnknownLocation();

        auto baseExpr = Fortran::semantics::AnalyzeExpr(context, x);
        fir::ExtendedValue variable = converter.genExprAddr(*baseExpr, stmtCtx, &loc);
        auto type = fir::getBase(variable).getType();
        if (type.isa<fir::BaseBoxType>()) {
          if (fir::isAllocatableType(type))
            retOperands_.push_back(fir::HeapType::get(fir::dyn_cast_ptrOrBoxEleTy(type)));
          else if (fir::isPointerType(type))
            retOperands_.push_back(fir::PointerType::get(fir::dyn_cast_ptrOrBoxEleTy(type)));
          else
            retOperands_.push_back(fir::ReferenceType::get(fir::dyn_cast_ptrOrBoxEleTy(type)));
        } else {
          retOperands_.push_back(type);
        }

        for (size_t i = 0; i < getNumDims(type); ++i)
          AddDimStartEnd();
      }

      Walk(x.base);
      return false;
    }
    bool Pre(const Fortran::parser::ArrayElement &x) {
      Walk(x.base);
      Walk(x.subscripts);
      return false;
    }
    void Post(const Fortran::parser::Name &name) {
      // We visit the base of all the expressions, so the first
      // decl value is the dep base.
      if (!baseOperand_) {
        baseOperand_ = fir::getBase(converter.getSymbolExtendedValue(*name.symbol));

        // TODO: location
        auto loc = converter.genUnknownLocation();
        auto baseExpr = Fortran::semantics::AnalyzeExpr(context, name);
        fir::ExtendedValue variable = converter.genExprAddr(*baseExpr, stmtCtx, &loc);
        if (retOperands_.empty()) {
          auto type = fir::getBase(variable).getType();
          if (type.isa<fir::BaseBoxType>()) {
            if (fir::isAllocatableType(type))
              retOperands_.push_back(fir::HeapType::get(fir::dyn_cast_ptrOrBoxEleTy(type)));
            else if (fir::isPointerType(type))
              retOperands_.push_back(fir::PointerType::get(fir::dyn_cast_ptrOrBoxEleTy(type)));
            else
              retOperands_.push_back(fir::ReferenceType::get(fir::dyn_cast_ptrOrBoxEleTy(type)));
          } else {
            retOperands_.push_back(type);
          }

          for (size_t i = 0; i < getNumDims(type); ++i)
            AddDimStartEnd();
        }
      }
      FillTypeVLASizes(*name.symbol);
      paramSymbols_.insert(*name.symbol);
    }

    mlir::Value baseOperand() const { return baseOperand_; }
    llvm::ArrayRef<Fortran::semantics::SymbolRef> paramSymbols() const { return paramSymbols_.getArrayRef(); }
    llvm::ArrayRef<mlir::Type> retOperands() const { return retOperands_; }

    private:
      void FillTypeVLASizes(Fortran::semantics::SymbolRef sym) {
        if (const auto *details{sym->detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
          for (const auto &shapeSpec : details->shape()) {
            struct ExprSymbolGathering : public Fortran::evaluate::AnyTraverse<ExprSymbolGathering> {
              using Base = AnyTraverse<ExprSymbolGathering>;
              ExprSymbolGathering(llvm::SetVector<Fortran::semantics::SymbolRef> &paramSymbols)
                : Base{*this}, paramSymbols_(paramSymbols) {}
              using Base::operator();
              bool operator()(const Fortran::semantics::Symbol &x) {
                paramSymbols_.insert(x);
                return true;
              }
              llvm::SetVector<Fortran::semantics::SymbolRef> &paramSymbols_;
            };
            if (const auto &lb{shapeSpec.lbound().GetExplicit()}) {
              if (!IsConstantExpr(*lb)) {
                auto expr = Fortran::semantics::SomeExpr{*lb};
                ExprSymbolGathering{paramSymbols_}(expr);
              }
            }
            if (const auto &ub{shapeSpec.ubound().GetExplicit()}) {
              if (!IsConstantExpr(*ub)) {
                auto expr = Fortran::semantics::SomeExpr{*ub};
                ExprSymbolGathering{paramSymbols_}(expr);
              }
            }
          }
        }
      }

      size_t getNumDims(mlir::Type type) {
        unsigned ndims = 1;
        if (auto baseType = fir::dyn_cast_ptrOrBoxEleTy(type))
          type = baseType;
        if (auto arrayType = type.dyn_cast<fir::SequenceType>())
          ndims = arrayType.getDimension();
        return ndims;
      }

      void AddDimStartEnd() {
        auto &firOpBuilder = converter.getFirOpBuilder();
        // TODO: is this the correct type?
        retOperands_.push_back(firOpBuilder.getI64Type());
        retOperands_.push_back(firOpBuilder.getI64Type());
        retOperands_.push_back(firOpBuilder.getI64Type());
      }

      Fortran::lower::AbstractConverter &converter;
      Fortran::semantics::SemanticsContext &context;
      Fortran::lower::StatementContext &stmtCtx;

      mlir::Value baseOperand_;
      // Used by regular variables involved in the expression
      llvm::SetVector<Fortran::semantics::SymbolRef> paramSymbols_;
      // TODO: is this the correct type?
      llvm::SmallVector<mlir::Type, 3> retOperands_;
  };
} // namespace

namespace {
  class OSSDependVisitor {
  public:
    explicit OSSDependVisitor(
        Fortran::lower::AbstractConverter &converter,
        Fortran::semantics::SemanticsContext &context,
        Fortran::lower::StatementContext &stmtCtx)
      : converter(converter), context(context), stmtCtx(stmtCtx),
        ossType(converter.getFirOpBuilder().getI64Type()) {}

    template <typename A> void Walk(const A &x) { Fortran::parser::Walk(x, *this); }
    template <typename A> bool Pre(const A &) { return true; }
    template <typename A> void Post(const A &) {}
    bool Pre(const Fortran::parser::OSSObject &x) {
      elemSize = x.elemSize;
      return true;
    }
    bool Pre(const Fortran::parser::StructureComponent &x) {
      auto &firOpBuilder = converter.getFirOpBuilder();
      // TODO: location
      auto loc = converter.genUnknownLocation();

      auto baseExpr = Fortran::semantics::AnalyzeExpr(context, x);
      fir::ExtendedValue exv = converter.genExprAddr(*baseExpr, stmtCtx, &loc);
      auto baseValue = fir::getBase(exv);
      // Pointers and assumed-share are BoxValues that need special handling
      if (const auto *box = exv.getBoxOf<fir::BoxValue>()) {
        exv = fir::factory::readBoxValue(firOpBuilder, loc, *box);
        baseValue = fir::getBase(exv);
      }
      returnList_.push_back(baseValue);

      const Fortran::semantics::Symbol *sym = Fortran::evaluate::GetLastSymbol(baseExpr);
      ProcessSymbol(*sym, exv);
      return false;
    }

    /// Compute extent from lower and upper bound.
    // NOTE: Borrowed from ConvertVariable.cpp
    static mlir::Value computeExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                                     mlir::Type type, mlir::Value lb, mlir::Value ub) {
      // Let the folder deal with the common `ub - <const> + 1` case.
      auto diff = builder.create<mlir::arith::SubIOp>(loc, type, ub, lb);
      mlir::Value one = builder.createIntegerConstant(loc, type, 1);
      auto rawExtent = builder.create<mlir::arith::AddIOp>(loc, type, diff, one);
      return fir::factory::genMaxWithZero(builder, loc, rawExtent);
    }

    bool Pre(const Fortran::parser::ArrayElement &x) {
      // TODO: location
      auto loc = converter.genUnknownLocation();
      auto &firOpBuilder = converter.getFirOpBuilder();

      auto baseExpr = Fortran::semantics::AnalyzeExpr(context, x.base);
      fir::ExtendedValue exv = converter.genExprAddr(*baseExpr, stmtCtx, &loc);
      auto baseValue = fir::getBase(exv);
      // Pointers and assumed-share are BoxValues that need special handling
      if (const auto *box = exv.getBoxOf<fir::BoxValue>()) {
        exv = fir::factory::readBoxValue(firOpBuilder, loc, *box);
        baseValue = fir::getBase(exv);
      }
      returnList_.push_back(baseValue);

      const Fortran::semantics::Symbol *sym = Fortran::evaluate::GetLastSymbol(baseExpr);
      Fortran::lower::BoxAnalyzer sba;
      sba.analyze(*sym);

      llvm::SmallVector<mlir::Value, 2> sizes;
      llvm::SmallVector<mlir::Value, 2> lBounds;
      llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 2> subscripts;
      auto subscriptIt = x.subscripts.begin();
      for (size_t i = 0; i < x.subscripts.size(); ++i) {
        mlir::Value lbDecl;
        mlir::Value ubDecl;
        mlir::Value assumedSizeSize;
        if (sba.isStaticArray()) {
          lbDecl = firOpBuilder.createIntegerConstant(loc, ossType, sba.staticLBound()[i]);
          sizes.push_back(firOpBuilder.createIntegerConstant(loc, ossType, sba.staticShape()[i]));
        } else if (sba.dynamicBound()[i]->ubound().isExplicit()) {
          auto lbExpr = sba.dynamicBound()[i]->lbound().GetExplicit();
          auto ubExpr = sba.dynamicBound()[i]->ubound().GetExplicit();
          lbDecl = fir::getBase(converter.genExprValue(Fortran::semantics::SomeExpr{*lbExpr}, stmtCtx));
          ubDecl = fir::getBase(converter.genExprValue(Fortran::semantics::SomeExpr{*ubExpr}, stmtCtx));
          sizes.push_back(computeExtent(firOpBuilder, loc, ossType, lbDecl, ubDecl));
        } else if (sba.dynamicBound()[i]->ubound().isStar()) {
          auto lbExpr = sba.dynamicBound()[i]->lbound().GetExplicit();
          lbDecl = fir::getBase(converter.genExprValue(Fortran::semantics::SomeExpr{*lbExpr}, stmtCtx));
          assumedSizeSize = lbDecl;
        } else if (sba.dynamicBound()[i]->ubound().isColon()) {
          mlir::IndexType idxTy = firOpBuilder.getIndexType();
          mlir::Value one = firOpBuilder.createIntegerConstant(loc, idxTy, 1);
          mlir::Value lb = fir::factory::readLowerBound(firOpBuilder, loc, exv, i, one);
          lbDecl = firOpBuilder.createConvert(loc, ossType, lb);
          mlir::Value extent = fir::factory::readExtent(firOpBuilder, loc, exv, i);
          extent = firOpBuilder.createConvert(loc, ossType, extent);
          ubDecl = firOpBuilder.create<mlir::arith::AddIOp>(loc, extent, lbDecl);
          ubDecl = firOpBuilder.create<mlir::arith::SubIOp>(loc, ubDecl, firOpBuilder.createIntegerConstant(loc, ossType, 1));
          sizes.push_back(computeExtent(firOpBuilder, loc, ossType, lbDecl, ubDecl));
        } else {
          llvm_unreachable("unexpected array type");
        }
        lBounds.push_back(lbDecl);

        if (auto *intExpr = std::get_if<Fortran::parser::IntExpr>(&subscriptIt->u)) {
          mlir::Value lb = fir::getBase(converter.genExprValue(
            *Fortran::semantics::AnalyzeExpr(context, intExpr->thing.value()), stmtCtx));
          lb = firOpBuilder.createConvert(loc, ossType, lb);
          subscripts.emplace_back(lb, lb);
        }
        if (auto *triplet{std::get_if<Fortran::parser::SubscriptTriplet>(&subscriptIt->u)}) {
          mlir::Value lb;
          if (!std::get<0>(triplet->t)) {
            lb = lbDecl;
          } else {
            lb = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(std::get<0>(triplet->t)), stmtCtx));
            lb = firOpBuilder.createConvert(loc, ossType, lb);
          }

          mlir::Value ub;
          if (!std::get<1>(triplet->t)) {
            if (sba.isStaticArray()) {
              ub = firOpBuilder.createIntegerConstant(loc, ossType, sba.staticShape()[i] + sba.staticLBound()[i] - 1);
            } else if (sba.dynamicBound()[i]->ubound().isExplicit()) {
              ub = ubDecl;
            } else if (sba.dynamicBound()[i]->ubound().isColon()) {
              ub = ubDecl;
            } else if (sba.dynamicBound()[i]->ubound().isStar()) {
              llvm_unreachable("assumed-size must have a section upper bound");
            }
          } else {
            ub = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(std::get<1>(triplet->t)), stmtCtx));
            ub = firOpBuilder.createConvert(loc, ossType, ub);
            // If we're in an assumed-size and have a section
            // set the size as the extent computed from it
            if (!sba.isStaticArray() && sba.dynamicBound()[i]->ubound().isStar())
              assumedSizeSize = computeExtent(firOpBuilder, loc, ossType, lb, ub);
          }
          subscripts.emplace_back(lb, ub);
        }

        // Push the size of the assumed-size
        // which can be the extent from the bounds
        // or 1
        if (assumedSizeSize)
          sizes.push_back(assumedSizeSize);

        subscriptIt++;
      }
      ComputeDepRanges(sizes, lBounds, subscripts);
      return false;
    }
    void Post(const Fortran::parser::Name &name) {
      // TODO: location
      auto loc = converter.genUnknownLocation();
      auto &firOpBuilder = converter.getFirOpBuilder();

      auto baseExpr = Fortran::semantics::AnalyzeExpr(context, name);
      fir::ExtendedValue exv = converter.genExprAddr(*baseExpr, stmtCtx, &loc);
      auto baseValue = fir::getBase(exv);
      // Pointers and assumed-share are BoxValues that need special handling
      if (const auto *box = exv.getBoxOf<fir::BoxValue>()) {
        exv = fir::factory::readBoxValue(firOpBuilder, loc, *box);
        baseValue = fir::getBase(exv);
      }
      returnList_.push_back(baseValue);

      const Fortran::semantics::Symbol *sym = Fortran::evaluate::GetLastSymbol(baseExpr);
      ProcessSymbol(*sym, exv);
    }

    llvm::ArrayRef<mlir::Value> returnList() const { return returnList_; }

    private:
      void ProcessSymbol(Fortran::semantics::SymbolRef sym, fir::ExtendedValue exv) {
        auto &firOpBuilder = converter.getFirOpBuilder();

        // TODO: location
        auto loc = converter.genUnknownLocation();
        mlir::IndexType idxTy = firOpBuilder.getIndexType();

        llvm::SmallVector<mlir::Value, 2> sizes;
        llvm::SmallVector<mlir::Value, 2> lBounds;
        llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 2> subscripts;

        Fortran::lower::BoxAnalyzer sba;
        sba.analyze(sym);
        if (sba.isArray()) {
          if (sba.isStaticArray()) {
            llvm::ArrayRef<int64_t> staticShape = sba.staticShape();
            llvm::ArrayRef<int64_t> staticLBound = sba.staticLBound();
            for (size_t i = 0; i < staticShape.size(); ++i) {
              mlir::Value size = firOpBuilder.createIntegerConstant(loc, ossType, staticShape[i]);
              sizes.push_back(size);
              mlir::Value lbVal = firOpBuilder.createIntegerConstant(loc, ossType, staticLBound[i]);
              lBounds.push_back(lbVal);
              subscripts.emplace_back(lbVal, firOpBuilder.createIntegerConstant(loc, ossType, staticShape[i] + staticLBound[i] - 1));
            }
          } else {
            // TODO, los vla si se hacen asi, pero los (:) necesitan acceder al box
            llvm::ArrayRef<const Fortran::semantics::ShapeSpec *> dynamicBound = sba.dynamicBound();

            for (size_t i = 0; i < dynamicBound.size(); ++i) {
              const Fortran::semantics::ShapeSpec *spec = dynamicBound[i];

              mlir::Value lbVal;
              if (auto lbExpr = spec->lbound().GetExplicit()) {
                lbVal = fir::getBase(converter.genExprValue(Fortran::semantics::SomeExpr{*lbExpr}, stmtCtx));
              } else if (spec->lbound().isColon()) {
                mlir::Value one = firOpBuilder.createIntegerConstant(loc, idxTy, 1);
                mlir::Value lb = fir::factory::readLowerBound(firOpBuilder, loc, exv, i, one);
                lbVal = firOpBuilder.createConvert(loc, ossType, lb);
              }

              mlir::Value ubVal;
              if (auto ubExpr = spec->ubound().GetExplicit()) {
                ubVal = fir::getBase(converter.genExprValue(Fortran::semantics::SomeExpr{*ubExpr}, stmtCtx));
              } else if (spec->ubound().isColon()) {
                mlir::Value extent = fir::factory::readExtent(firOpBuilder, loc, exv, i);
                extent = firOpBuilder.createConvert(loc, ossType, extent);
                ubVal = firOpBuilder.create<mlir::arith::AddIOp>(loc, extent, lbVal);
                ubVal = firOpBuilder.create<mlir::arith::SubIOp>(loc, ubVal, firOpBuilder.createIntegerConstant(loc, ossType, 1));
              }

              mlir::Value size = firOpBuilder.createIntegerConstant(loc, ossType, 1);
              size = firOpBuilder.create<mlir::arith::AddIOp>(loc, size, ubVal);
              size = firOpBuilder.create<mlir::arith::SubIOp>(loc, size, lbVal);
              sizes.push_back(size);

              lBounds.push_back(lbVal);
              subscripts.emplace_back(lbVal, ubVal);
            }
          }
        } else {
          mlir::Value c1 = firOpBuilder.createIntegerConstant(loc, ossType, 1);
          sizes.push_back(c1);
          lBounds.push_back(c1);
          subscripts.emplace_back(c1, c1);
        }
        ComputeDepRanges(sizes, lBounds, subscripts);
      }

      void ComputeDepRanges(
          llvm::ArrayRef<mlir::Value> sizes,
          llvm::ArrayRef<mlir::Value> lBounds,
          llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> subscripts) {
        assert(!sizes.empty());
        assert(sizes.size() == lBounds.size());
        assert(lBounds.size() == subscripts.size());

        auto &firOpBuilder = converter.getFirOpBuilder();

        // TODO: location
        auto loc = converter.genUnknownLocation();

        mlir::Value c1 = firOpBuilder.createIntegerConstant(loc, ossType, 1);
        mlir::Value elemSizeBytes = firOpBuilder.createIntegerConstant(loc, ossType, elemSize);
        for (size_t i = 0; i < sizes.size(); ++i) {
          returnList_.push_back(sizes[i]);

          mlir::Value depLB = subscripts[i].first;
          depLB = firOpBuilder.create<mlir::arith::SubIOp>(loc, depLB, lBounds[i]);
          returnList_.push_back(depLB);

          mlir::Value depUB = subscripts[i].second;
          depUB = firOpBuilder.create<mlir::arith::SubIOp>(loc, depUB, lBounds[i]);
          depUB = firOpBuilder.create<mlir::arith::AddIOp>(loc, depUB, c1);
          returnList_.push_back(depUB);
        }
        // Convert the first size/start/end to bytes
        returnList_[1] = firOpBuilder.create<mlir::arith::MulIOp>(loc, returnList_[1], elemSizeBytes);
        returnList_[2] = firOpBuilder.create<mlir::arith::MulIOp>(loc, returnList_[2], elemSizeBytes);
        returnList_[3] = firOpBuilder.create<mlir::arith::MulIOp>(loc, returnList_[3], elemSizeBytes);
      }

      Fortran::lower::AbstractConverter &converter;
      Fortran::semantics::SemanticsContext &context;
      Fortran::lower::StatementContext &stmtCtx;
      llvm::SmallVector<mlir::Value, 4> returnList_;
      const mlir::Type ossType;
      size_t elemSize;
  };
} // namespace

namespace {
  class OSSLoopBoundsVisitor {
  public:
    explicit OSSLoopBoundsVisitor(
        Fortran::lower::AbstractConverter &converter,
        Fortran::lower::StatementContext &stmtCtx)
      : converter(converter), stmtCtx(stmtCtx) {}

    template <typename A> void Walk(const A &x) { Fortran::parser::Walk(x, *this); }
    template <typename A> bool Pre(const A &) { return true; }
    template <typename A> void Post(const A &) {}
    bool Pre(const Fortran::parser::DoConstruct &x) {
      const auto &control{x.GetLoopControl()};
      const auto &bounds{std::get<Fortran::parser::LoopControl::Bounds>(control->u)};
      const Fortran::semantics::Symbol *symbol{bounds.name.thing.symbol};
      indVarOperand_ = addOperands(converter, *symbol);
      lowerBoundOperand_ =
        fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(bounds.lower), stmtCtx));
      upperBoundOperand_ =
        fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(bounds.upper), stmtCtx));
      auto &firOpBuilder = converter.getFirOpBuilder();
      auto currentLocation = converter.getCurrentLocation();
      if (bounds.step) {
        stepOperand_ =
          fir::getBase(
            converter.genExprValue(*Fortran::semantics::GetExpr(bounds.step), stmtCtx));
      } else {
        stepOperand_ =
          firOpBuilder.createIntegerConstant(currentLocation, converter.genType(*symbol), 1);
      }
      // NOTE: 1 means LE (<=) loop type
      loopTypeOperand_ =
        firOpBuilder.createIntegerConstant(currentLocation, firOpBuilder.getI64Type(), 1);

      return false;
    }

    mlir::Value indVarOperand() const { return indVarOperand_; }
    mlir::Value lowerBoundOperand() const { return lowerBoundOperand_; }
    mlir::Value upperBoundOperand() const { return upperBoundOperand_; }
    mlir::Value stepOperand() const { return stepOperand_; }
    mlir::Value loopTypeOperand() const { return loopTypeOperand_; }

  private:
    Fortran::lower::AbstractConverter &converter;
    Fortran::lower::StatementContext &stmtCtx;
    mlir::Value indVarOperand_;
    mlir::Value lowerBoundOperand_;
    mlir::Value upperBoundOperand_;
    mlir::Value stepOperand_;
    mlir::Value loopTypeOperand_;
  };
} // namespace

static mlir::Value getComputeDep(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &context,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::semantics::Scope &scope,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::OSSObject &ossObject) {
  OSSDependInfoGathering dependInfoGathering(converter, context, stmtCtx);
  dependInfoGathering.Walk(ossObject);

  mlir::Value baseOperand = dependInfoGathering.baseOperand();
  const auto &paramSymbols = dependInfoGathering.paramSymbols();
  llvm::ArrayRef<mlir::Type> retOperands = dependInfoGathering.retOperands();
  llvm::SmallVector<mlir::Type, 4> paramTypes;
  llvm::SmallVector<mlir::Value, 4> paramValues;
  for (const auto &sym : paramSymbols) {
    mlir::Value val = addOperands(converter, sym);
    paramTypes.push_back(val.getType());
    paramValues.push_back(val);
  }

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  auto ty = mlir::FunctionType::get(
    firOpBuilder.getContext(), paramTypes, retOperands);

  std::string funcName = "compute.dep";
  static size_t instance = 0;
  funcName += std::to_string(instance++);

  auto func = firOpBuilder.createFunction(currentLocation, funcName.c_str(), ty);
  func.setVisibility(mlir::SymbolTable::Visibility::Public);
  func.addEntryBlock();
  auto &block = func.getRegion().back();

  auto insertPt = firOpBuilder.saveInsertionPoint();

  firOpBuilder.setInsertionPointToStart(&block);

  Fortran::lower::SymMap localSymbols;

  // Map symbols to function arguments
  for (size_t i = 0; i < paramSymbols.size(); ++i) {
    fir::ExtendedValue exv = converter.getSymbolExtendedValue(paramSymbols[i]);
    if (auto *box = exv.getBoxOf<fir::MutableBoxValue>()) {
      auto loc = converter.genUnknownLocation();

      auto rank = box->rank();
      const auto &mutableProperties = box->getMutableProperties();

      fir::MutableProperties newMutablePropertiesDst;

      if (box->isDescribedByVariables()) {
        newMutablePropertiesDst.addr = firOpBuilder.createTemporary(loc, box->getMemTy(), "addr.dst");
        for (decltype(rank) dim = 0; dim < rank; ++dim) {
          if (mutableProperties.lbounds[dim]) {
            newMutablePropertiesDst.lbounds.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "lb.dst"));
          }
          if (mutableProperties.extents[dim]) {
            newMutablePropertiesDst.extents.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "extent.dst"));
          }
        }
      }

      auto newBoxDst = fir::MutableBoxValue(
        func.front().getArguments()[i], box->nonDeferredLenParams(), newMutablePropertiesDst);

      // 2899b42f075fe8c016efb111b023ec6e8742642e dice que pushear un scope va a hacer
      // que no se encuentre el iterador del bucle al emitir la op de oss.taskloop
      localSymbols.addAllocatableOrPointer(paramSymbols[i], newBoxDst);
      fir::factory::syncMutableBoxFromIRBox(firOpBuilder, loc, newBoxDst);
    } else if (auto *box = exv.getBoxOf<fir::BoxValue>()) {
      localSymbols.addBoxSymbol(
        paramSymbols[i], func.front().getArguments()[i], box->getLBounds(), box->getExplicitExtents(), box->getExplicitParameters());
    } else {
      localSymbols.addSymbol(paramSymbols[i], func.front().getArguments()[i]);
    }
  }

  for (const auto &var : Fortran::lower::pft::getScopeVariableList(scope)) {
    const Fortran::semantics::Symbol &sym = var.getSymbol();
    for (size_t i = 0; i < paramSymbols.size(); ++i) {
      if (paramSymbols[i] == sym) {
        Fortran::lower::mapSymbolAttributes(
          converter, var, localSymbols, stmtCtx, func.front().getArguments()[i]);
        break;
      }
    }
  }

  // llvm::dbgs() << "stack size: " << localSymbols.symbolMapStack.size() << "\n";

  converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

  OSSDependVisitor dependVisitor(converter, context, stmtCtx);
  dependVisitor.Walk(ossObject);

  llvm::ArrayRef<mlir::Value> returnList = dependVisitor.returnList();

  // Ensure the block is well-formed.
  firOpBuilder.create<mlir::func::ReturnOp>(currentLocation, returnList);

  converter.getLocalSymbols().popScope();

  firOpBuilder.restoreInsertionPoint(insertPt);

  auto op = firOpBuilder.create<mlir::oss::DepOp>(
    currentLocation, firOpBuilder.getI32Type(), baseOperand,
    func.getSymName(), paramValues).getResult();
  return op;
}

namespace {
  class OSSClausesVisitor {
  public:
    explicit OSSClausesVisitor(
        Fortran::lower::AbstractConverter &converter,
        Fortran::semantics::SemanticsContext &context,
        Fortran::lower::pft::Evaluation &eval,
        const Fortran::semantics::Scope &scope,
        Fortran::lower::StatementContext &stmtCtx)
      : converter(converter), context(context), eval(eval), scope(scope), stmtCtx(stmtCtx) {}

    explicit OSSClausesVisitor(
        Fortran::lower::AbstractConverter &converter,
        Fortran::semantics::SemanticsContext &context,
        Fortran::lower::pft::Evaluation &eval,
        const Fortran::semantics::Scope &scope,
        Fortran::lower::StatementContext &stmtCtx,
        const Fortran::lower::ImplicitDSAs &implicitDSAs)
      : converter(converter), context(context), eval(eval), scope(scope), stmtCtx(stmtCtx) {

      for (const auto &sym : implicitDSAs.sharedList) {
        addOperandsBoxAware(sym, sharedClauseOperands_);
        addAdditionalInfo(sym);
      }
      for (const auto &sym : implicitDSAs.privateList) {
        addOperandsBoxAware(sym, privateClauseOperands_);
        addAdditionalInfo(sym, /*InitNeeded=*/true);
      }
      for (const auto &sym : implicitDSAs.firstprivateList) {
        addOperandsBoxAware(sym, firstprivateClauseOperands_);
        addAdditionalInfo(sym, /*InitNeeded=*/false, /*CopyNeeded=*/true);
      }
    }

    // ShouldPack is needed in private variables. We pack the box there to
    // have the updated values, do the copy and unpack the values again
    // In shared variables we need to annotate both the box and the
    // unpacked values
    void addOperandsBoxAware(
        Fortran::lower::SymbolRef sym, llvm::SetVector<mlir::Value> &list, bool ShouldPack = true) {
      auto &firOpBuilder = converter.getFirOpBuilder();
      const auto variable = converter.getSymbolAddress(sym);
      if (variable) {
        fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym);
        if (auto *box = exv.getBoxOf<fir::MutableBoxValue>()) {
          if (ShouldPack) {
            auto loc = converter.genUnknownLocation();
            list.insert(fir::factory::getMutableIRBox(firOpBuilder, loc, *box));
            return;
          } else {
            auto rank = box->rank();
            const auto &mutableProperties = box->getMutableProperties();

            list.insert(fir::getBase(exv));
            if (box->isDescribedByVariables()) {
              // TODO: Idk if we have to check if the values exists
              list.insert(mutableProperties.addr);
              for (decltype(rank) dim = 0; dim < rank; ++dim) {
                if (mutableProperties.lbounds[dim])
                  list.insert(mutableProperties.lbounds[dim]);
                if (mutableProperties.extents[dim])
                  list.insert(mutableProperties.extents[dim]);
              }
            }
            return;
          }
        } else {
          list.insert(variable);
          return;
        }
      } else {
        if (const auto *details =
                sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
          converter.copySymbolBinding(details->symbol(), sym);
          list.insert(converter.getSymbolAddress(details->symbol()));
          return;
        }
      }
      llvm_unreachable("addOperands");
    }

    void getCopyMutableBoxValueGen(
        Fortran::lower::SymbolRef sym,
        fir::ExtendedValue exv,
        fir::FirOpBuilder &firOpBuilder,
        mlir::Location loc,
        Fortran::lower::SymMap &localSymbols,
        mlir::Value dst, mlir::Value src, bool DoNotInitialize) {

      auto *box = exv.getBoxOf<fir::MutableBoxValue>();

      auto rank = box->rank();
      const auto &mutableProperties = box->getMutableProperties();

      fir::MutableProperties newMutablePropertiesSrc;
      fir::MutableProperties newMutablePropertiesDst;

      if (box->isDescribedByVariables()) {
        newMutablePropertiesSrc.addr = firOpBuilder.createTemporary(loc, box->getMemTy(), "addr.src");
        newMutablePropertiesDst.addr = firOpBuilder.createTemporary(loc, box->getMemTy(), "addr.dst");
        for (decltype(rank) dim = 0; dim < rank; ++dim) {
          if (mutableProperties.lbounds[dim]) {
            newMutablePropertiesSrc.lbounds.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "lb.src"));
            newMutablePropertiesDst.lbounds.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "lb.dst"));
          }
          if (mutableProperties.extents[dim]) {
            newMutablePropertiesSrc.extents.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "extent.src"));
            newMutablePropertiesDst.extents.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "extent.dst"));
          }
        }
      }

      auto newBoxSrc = fir::MutableBoxValue(
        src, box->nonDeferredLenParams(), newMutablePropertiesSrc);
      auto newBoxDst = fir::MutableBoxValue(
        dst, box->nonDeferredLenParams(), newMutablePropertiesDst);

      // 2899b42f075fe8c016efb111b023ec6e8742642e dice que pushear un scope va a hacer
      // que no se encuentre el iterador del bucle al emitir la op de oss.taskloop
      localSymbols.addAllocatableOrPointer(sym, newBoxSrc);
      assert(sym->getOssAdditionalSym() && "Expected to have a temporal symbol to emit the expression");
      localSymbols.addAllocatableOrPointer(*sym->getOssAdditionalSym(), newBoxDst);
      converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

      if (box->isAllocatable()) {
        // Sync described variables if it is the case
        fir::factory::syncMutableBoxFromIRBox(firOpBuilder, loc, newBoxSrc);
        fir::factory::syncMutableBoxFromIRBox(firOpBuilder, loc, newBoxDst);
        mlir::Value isAllocated = fir::factory::genIsAllocatedOrAssociatedTest(firOpBuilder, loc, newBoxSrc);
        firOpBuilder.genIfThen(loc, isAllocated)
            .genThen([&]() {
              emitOSSCopyExpr(converter, sym, stmtCtx, DoNotInitialize);
            })
            .end();
        // Pack again the modified described variables in the box if it is the case
        fir::factory::getMutableIRBox(firOpBuilder, loc, newBoxDst);
      } else {
        fir::ExtendedValue newBoxSrcLoad = fir::factory::genMutableBoxRead(firOpBuilder, loc, newBoxSrc);
        fir::factory::associateMutableBox(firOpBuilder, loc, newBoxDst, newBoxSrcLoad, std::nullopt);
      }
    }

    void getCopyDerivedGen(
        Fortran::lower::SymbolRef sym,
        fir::ExtendedValue exv,
        fir::FirOpBuilder &firOpBuilder,
        mlir::Location loc,
        Fortran::lower::SymMap &localSymbols,
        mlir::Value dst, mlir::Value src) {

      // 2899b42f075fe8c016efb111b023ec6e8742642e dice que pushear un scope va a hacer
      // que no se encuentre el iterador del bucle al emitir la op de oss.taskloop
      localSymbols.addSymbol(sym, src);
      assert(sym->getOssAdditionalSym() && "Expected to have a temporal symbol to emit the expression");
      localSymbols.addSymbol(*sym->getOssAdditionalSym(), dst);
      converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

      // Assignment requires first an initialization
      mlir::Value TmpBox = firOpBuilder.createBox(loc, dst);
      fir::runtime::genDerivedTypeInitialize(firOpBuilder, loc, TmpBox);
      fir::factory::genRecordAssignment(firOpBuilder, loc, dst, src);
    }

    mlir::Value getCopyMutableBoxValue(Fortran::lower::SymbolRef sym, bool DoNotInitialize) {
      return getCopyImpl</*IsMutable=*/true>(sym, DoNotInitialize);
    }

    mlir::Value getCopyDerived(Fortran::lower::SymbolRef sym) {
      return getCopyImpl</*IsMutable=*/false>(sym);
    }

    template<bool IsMutable>
    mlir::Value getCopyImpl(Fortran::lower::SymbolRef sym, bool DoNotInitialize=false) {
      auto &firOpBuilder = converter.getFirOpBuilder();

      llvm::SmallVector<mlir::Type, 4> paramTypes;

      // Return whatever
      llvm::SmallVector<mlir::Type, 4> retOperands;
      retOperands.push_back(firOpBuilder.getI32Type());

      fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym);
      mlir::Value base = fir::getBase(exv);
      paramTypes.push_back(base.getType());
      paramTypes.push_back(base.getType());
      paramTypes.push_back(firOpBuilder.getI64Type());

      auto currentLocation = converter.getCurrentLocation();

      auto ty = mlir::FunctionType::get(
        firOpBuilder.getContext(), paramTypes, retOperands);

      std::string funcName = "compute.copy";
      static size_t instance = 0;
      funcName += std::to_string(instance++);

      auto func = firOpBuilder.createFunction(currentLocation, funcName.c_str(), ty);
      func.setVisibility(mlir::SymbolTable::Visibility::Public);
      func.addEntryBlock();
      auto &block = func.getRegion().back();

      auto insertPt = firOpBuilder.saveInsertionPoint();

      firOpBuilder.setInsertionPointToStart(&block);

      Fortran::lower::SymMap localSymbols;
      auto loc = converter.genUnknownLocation();

      if (IsMutable) {
        getCopyMutableBoxValueGen(
          sym, exv, firOpBuilder, loc, localSymbols, func.front().getArguments()[1], func.front().getArguments()[0], DoNotInitialize);
      } else {
        getCopyDerivedGen(
          sym, exv, firOpBuilder, loc, localSymbols, func.front().getArguments()[1], func.front().getArguments()[0]);
      }

      llvm::SmallVector<mlir::Value, 4> returnList;
      returnList.push_back(firOpBuilder.createIntegerConstant(currentLocation, firOpBuilder.getI32Type(), 1));
      // // Ensure the block is well-formed.
      firOpBuilder.create<mlir::func::ReturnOp>(currentLocation, returnList);

      converter.getLocalSymbols().popScope();

      firOpBuilder.restoreInsertionPoint(insertPt);

      auto op = firOpBuilder.create<mlir::oss::CopyOp>(
        currentLocation, firOpBuilder.getI32Type(), base,
        func.getSymName()).getResult();
      return op;
    }

    void getInitMutableBoxValueGen(
        Fortran::lower::SymbolRef sym,
        fir::ExtendedValue exv,
        fir::FirOpBuilder &firOpBuilder,
        mlir::Location loc,
        Fortran::lower::SymMap &localSymbols,
        mlir::Value dst) {

      auto *box = exv.getBoxOf<fir::MutableBoxValue>();

      auto rank = box->rank();
      const auto &mutableProperties = box->getMutableProperties();

      fir::MutableProperties newMutablePropertiesDst;

      if (box->isDescribedByVariables()) {
        newMutablePropertiesDst.addr = firOpBuilder.createTemporary(loc, box->getMemTy(), "addr.dst");
        for (decltype(rank) dim = 0; dim < rank; ++dim) {
          if (mutableProperties.lbounds[dim]) {
            newMutablePropertiesDst.lbounds.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "lb.dst"));
          }
          if (mutableProperties.extents[dim]) {
            newMutablePropertiesDst.extents.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "extent.dst"));
          }
        }
      }

      auto newBoxDst = fir::MutableBoxValue(
        dst, box->nonDeferredLenParams(), newMutablePropertiesDst);

      // 2899b42f075fe8c016efb111b023ec6e8742642e dice que pushear un scope va a hacer
      // que no se encuentre el iterador del bucle al emitir la op de oss.taskloop
      localSymbols.addAllocatableOrPointer(sym, newBoxDst);
      converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

      fir::factory::disassociateMutableBox(firOpBuilder, loc, newBoxDst);
      fir::factory::getMutableIRBox(firOpBuilder, loc, newBoxDst);
    }

    void getInitDerivedGen(
        Fortran::lower::SymbolRef sym,
        fir::ExtendedValue exv,
        fir::FirOpBuilder &firOpBuilder,
        mlir::Location loc,
        Fortran::lower::SymMap &localSymbols,
        mlir::Value dst) {

      converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

      mlir::Value TmpBox = firOpBuilder.createBox(loc, dst);
      fir::runtime::genDerivedTypeInitialize(firOpBuilder, loc, TmpBox);
    }

    mlir::Value getInitMutableBoxValue(Fortran::lower::SymbolRef sym) {
      return getInitImpl</*IsMutable=*/true>(sym);
    }

    mlir::Value getInitDerived(Fortran::lower::SymbolRef sym) {
      return getInitImpl</*IsMutable=*/false>(sym);
    }

    template<bool IsMutable>
    mlir::Value getInitImpl(Fortran::lower::SymbolRef sym) {
      auto &firOpBuilder = converter.getFirOpBuilder();

      llvm::SmallVector<mlir::Type, 4> paramTypes;

      // Return whatever
      llvm::SmallVector<mlir::Type, 4> retOperands;
      retOperands.push_back(firOpBuilder.getI32Type());

      fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym);
      mlir::Value base = fir::getBase(exv);
      paramTypes.push_back(base.getType());
      paramTypes.push_back(firOpBuilder.getI64Type());

      auto currentLocation = converter.getCurrentLocation();

      auto ty = mlir::FunctionType::get(
        firOpBuilder.getContext(), paramTypes, retOperands);

      std::string funcName = "compute.init";
      static size_t instance = 0;
      funcName += std::to_string(instance++);

      auto func = firOpBuilder.createFunction(currentLocation, funcName.c_str(), ty);
      func.setVisibility(mlir::SymbolTable::Visibility::Public);
      func.addEntryBlock();
      auto &block = func.getRegion().back();

      auto insertPt = firOpBuilder.saveInsertionPoint();

      firOpBuilder.setInsertionPointToStart(&block);

      Fortran::lower::SymMap localSymbols;
      auto loc = converter.genUnknownLocation();

      if (IsMutable) {
        getInitMutableBoxValueGen(
          sym, exv, firOpBuilder, loc, localSymbols, func.front().getArguments()[0]);
      } else {
        getInitDerivedGen(
          sym, exv, firOpBuilder, loc, localSymbols, func.front().getArguments()[0]);
      }

      llvm::SmallVector<mlir::Value, 4> returnList;
      returnList.push_back(firOpBuilder.createIntegerConstant(currentLocation, firOpBuilder.getI32Type(), 1));
      // // Ensure the block is well-formed.
      firOpBuilder.create<mlir::func::ReturnOp>(currentLocation, returnList);

      converter.getLocalSymbols().popScope();

      firOpBuilder.restoreInsertionPoint(insertPt);

      auto op = firOpBuilder.create<mlir::oss::CopyOp>(
        currentLocation, firOpBuilder.getI32Type(), base,
        func.getSymName()).getResult();
      return op;
    }

    void getDeinitMutableBoxValueGen(
        Fortran::lower::SymbolRef sym,
        fir::ExtendedValue exv,
        fir::FirOpBuilder &firOpBuilder,
        mlir::Location loc,
        Fortran::lower::SymMap &localSymbols,
        mlir::Value dst) {

      auto *box = exv.getBoxOf<fir::MutableBoxValue>();

      auto rank = box->rank();
      const auto &mutableProperties = box->getMutableProperties();

      fir::MutableProperties newMutablePropertiesDst;

      if (box->isDescribedByVariables()) {
        newMutablePropertiesDst.addr = firOpBuilder.createTemporary(loc, box->getMemTy(), "addr.dst");
        for (decltype(rank) dim = 0; dim < rank; ++dim) {
          if (mutableProperties.lbounds[dim]) {
            newMutablePropertiesDst.lbounds.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "lb.dst"));
          }
          if (mutableProperties.extents[dim]) {
            newMutablePropertiesDst.extents.push_back(
              firOpBuilder.createTemporary(loc, firOpBuilder.getIndexType(), "extent.dst"));
          }
        }
      }

      auto newBoxDst = fir::MutableBoxValue(
        dst, box->nonDeferredLenParams(), newMutablePropertiesDst);

      // 2899b42f075fe8c016efb111b023ec6e8742642e dice que pushear un scope va a hacer
      // que no se encuentre el iterador del bucle al emitir la op de oss.taskloop
      localSymbols.addAllocatableOrPointer(sym, newBoxDst);
      converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

      if (box->isAllocatable()) {
        fir::factory::syncMutableBoxFromIRBox(firOpBuilder, loc, newBoxDst);
        emitOSSDeinitExpr(converter, newBoxDst, loc);
        fir::factory::getMutableIRBox(firOpBuilder, loc, newBoxDst);
      } else {
        fir::factory::disassociateMutableBox(firOpBuilder, loc, newBoxDst);
      }
    }

    void getDeinitDerivedGen(
        Fortran::lower::SymbolRef sym,
        fir::ExtendedValue exv,
        fir::FirOpBuilder &firOpBuilder,
        mlir::Location loc,
        Fortran::lower::SymMap &localSymbols,
        mlir::Value dst) {

      converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

      mlir::Value TmpBox = firOpBuilder.createBox(loc, dst);
      fir::runtime::genDerivedTypeDestroy(firOpBuilder, loc, TmpBox);
    }

    mlir::Value getDeinitMutableBoxValue(Fortran::lower::SymbolRef sym) {
      return getDeinitImpl</*IsMutable=*/true>(sym);
    }

    mlir::Value getDeinitDerived(Fortran::lower::SymbolRef sym) {
      return getDeinitImpl</*IsMutable=*/false>(sym);
    }

    template<bool IsMutable>
    mlir::Value getDeinitImpl(Fortran::lower::SymbolRef sym) {
      auto &firOpBuilder = converter.getFirOpBuilder();

      llvm::SmallVector<mlir::Type, 4> paramTypes;

      // Return whatever
      llvm::SmallVector<mlir::Type, 4> retOperands;
      retOperands.push_back(firOpBuilder.getI32Type());

      fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym);
      mlir::Value base = fir::getBase(exv);
      paramTypes.push_back(base.getType());
      paramTypes.push_back(firOpBuilder.getI64Type());

      auto currentLocation = converter.getCurrentLocation();

      auto ty = mlir::FunctionType::get(
        firOpBuilder.getContext(), paramTypes, retOperands);

      std::string funcName = "compute.deinit";
      static size_t instance = 0;
      funcName += std::to_string(instance++);

      auto func = firOpBuilder.createFunction(currentLocation, funcName.c_str(), ty);
      func.setVisibility(mlir::SymbolTable::Visibility::Public);
      func.addEntryBlock();
      auto &block = func.getRegion().back();

      auto insertPt = firOpBuilder.saveInsertionPoint();

      firOpBuilder.setInsertionPointToStart(&block);

      Fortran::lower::SymMap localSymbols;
      auto loc = converter.genUnknownLocation();

      if (IsMutable) {
        getDeinitMutableBoxValueGen(
          sym, exv, firOpBuilder, loc, localSymbols, func.front().getArguments()[0]);
      } else {
        getDeinitDerivedGen(
          sym, exv, firOpBuilder, loc, localSymbols, func.front().getArguments()[0]);
      }

      llvm::SmallVector<mlir::Value, 4> returnList;
      returnList.push_back(firOpBuilder.createIntegerConstant(currentLocation, firOpBuilder.getI32Type(), 1));
      // // Ensure the block is well-formed.
      firOpBuilder.create<mlir::func::ReturnOp>(currentLocation, returnList);

      converter.getLocalSymbols().popScope();

      firOpBuilder.restoreInsertionPoint(insertPt);

      auto op = firOpBuilder.create<mlir::oss::CopyOp>(
        currentLocation, firOpBuilder.getI32Type(), base,
        func.getSymName()).getResult();
      return op;
    }

    // NOTE: borrowed from flang/lib/Optimizer/Builder/FIRBuilder.cpp
    /// Can the assignment of this record type be implement with a simple memory
    /// copy (it requires no deep copy or user defined assignment of components )?
    static bool recordTypeCanBeMemCopied(fir::RecordType recordType) {
      if (fir::hasDynamicSize(recordType))
        return false;
      for (auto [_, fieldType] : recordType.getTypeList()) {
        // Derived type component may have user assignment (so far, we cannot tell
        // in FIR, so assume it is always the case, TODO: get the actual info).
        if (fir::unwrapSequenceType(fieldType).isa<fir::RecordType>())
          return false;
        // Allocatable components need deep copy.
        if (auto boxType = fieldType.dyn_cast<fir::BoxType>())
          if (boxType.getEleTy().isa<fir::HeapType>())
            return false;
      }
      // Constant size components without user defined assignment and pointers can
      // be memcopied.
      return true;
    }

    // This function emits extra information (almost) independent
    // of the data-sharing.
    // InitNeeded: means the variable need to emit the initializer.
    // CopyNeeded: means the variable need to emit the copy initializer.
    void addAdditionalInfo(
        Fortran::lower::SymbolRef sym, bool InitNeeded = false,
        bool CopyNeeded = false) {
      const Fortran::semantics::Symbol &ultimate = sym->GetUltimate();
      auto currentLocation = converter.getCurrentLocation();
      auto &firOpBuilder = converter.getFirOpBuilder();
      Fortran::lower::BoxAnalyzer sba;
      sba.analyze(ultimate);
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> lBounds;
      fir::ExtendedValue exv = converter.getLocalSymbols().lookupSymbol(ultimate).toExtendedValue();
      if (sba.isArray()) {
        if (const auto *box = exv.getBoxOf<fir::ArrayBoxValue>()) {
          // assumed-size arrays have a first fir.underfined operation
          // as extent. We use this to indentify them to avoid
          // vlaOp creation, but emit captures
          bool IsPtr =
            !box->getExtents().empty() &&
            mlir::isa<fir::UndefOp>(box->getExtents().back().getDefiningOp());
          for (auto extent : box->getExtents()) {
            capturesOperands_.insert(extent);
            if (!sba.isStaticArray() && !IsPtr)
              extents.push_back(extent);
          }
          for (auto lb : box->getLBounds()) {
            capturesOperands_.insert(lb);
            if (!sba.isStaticArray() && !IsPtr)
              lBounds.push_back(lb);
          }
          // Only emit the vlaOp if has relevant information
          if (!(extents.empty() && lBounds.empty())) {
            auto vlaDim = firOpBuilder
                              .create<mlir::oss::VlaDimOp>(
                                  currentLocation, firOpBuilder.getI32Type(),
                                  addOperands(converter, ultimate),
                                  extents, lBounds)
                              .getResult();
            vlaDimsOperands_.push_back(vlaDim);
          }
        } else if (auto *box = exv.getBoxOf<fir::MutableBoxValue>()) {
          if (InitNeeded) {
            if (box->isAllocatable()) {
              auto copyOp = getCopyMutableBoxValue(sym, /*DoNotInitialize=*/true);
              auto deinitOp = getDeinitMutableBoxValue(sym);
              copyClauseOperands_.insert(copyOp);
              deinitClauseOperands_.insert(deinitOp);
              // FIXME: change the dsa since we will emit a "copy" function where
              // only we will allocate an uninitialized buffer if the orig variable
              // is allocated
              privateClauseOperands_.remove(fir::getBase(exv));
              firstprivateClauseOperands_.insert(fir::getBase(exv));
            } else {
              auto initOp = getInitMutableBoxValue(sym);
              auto deinitOp = getDeinitMutableBoxValue(sym);
              initClauseOperands_.insert(initOp);
              deinitClauseOperands_.insert(deinitOp);
            }
          } else if (CopyNeeded) {
            auto copyOp = getCopyMutableBoxValue(sym, /*DoNotInitialize=*/false);
            auto deinitOp = getDeinitMutableBoxValue(sym);
            copyClauseOperands_.insert(copyOp);
            deinitClauseOperands_.insert(deinitOp);
          }
        }
      } else if (sba.isTrivial()) {
        Fortran::evaluate::Expr expr{Fortran::evaluate::AsGenericExpr(sym).value()};
        if (auto exprType = expr.GetType()) {
          if (exprType->category() == Fortran::common::TypeCategory::Derived) {
            auto baseTy = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(exv).getType());
            assert(baseTy && "must be a memory type");
            auto recTy = baseTy.dyn_cast<fir::RecordType>();
            if (recordTypeCanBeMemCopied(recTy))
              return;
            if (InitNeeded) {
              auto initOp = getInitDerived(sym);
              auto deinitOp = getDeinitDerived(sym);
              initClauseOperands_.insert(initOp);
              deinitClauseOperands_.insert(deinitOp);
            } else if (CopyNeeded) {
              auto copyOp = getCopyDerived(sym);
              auto deinitOp = getDeinitDerived(sym);
              copyClauseOperands_.insert(copyOp);
              deinitClauseOperands_.insert(deinitOp);
            }
          }
        }
      }
    }

    void gatherClauseList(const Fortran::parser::OSSClauseList &clauseList) {
      for (const auto &clause : clauseList.v) {
        if (const auto &ifClause =
                std::get_if<Fortran::parser::OSSClause::If>(&clause.u)) {
          ifClauseOperand_ = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(ifClause->v), stmtCtx));
        } else if (const auto &finalClause =
                std::get_if<Fortran::parser::OSSClause::Final>(&clause.u)) {
          finalClauseOperand_ = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(finalClause->v), stmtCtx));
        } else if (const auto &costClause =
                       std::get_if<Fortran::parser::OSSClause::Cost>(
                           &clause.u)) {
          costClauseOperand_ = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(costClause->v), stmtCtx));
        } else if (const auto &priorityClause =
                       std::get_if<Fortran::parser::OSSClause::Priority>(
                           &clause.u)) {
          priorityClauseOperand_ = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(priorityClause->v), stmtCtx));
        } else if (const auto &defaultClause =
              std::get_if<Fortran::parser::OSSClause::Default>(&clause.u)) {
          const auto &ossDefaultClause{defaultClause->v};
          auto &firOpBuilder = converter.getFirOpBuilder();
          switch (ossDefaultClause.v) {
          case Fortran::parser::OSSDefaultClause::Type::Private:
            defaultClauseOperand_ =
              mlir::oss::ClauseDefaultAttr::get(firOpBuilder.getContext(), mlir::oss::ClauseDefault::defprivate);
            break;
          case Fortran::parser::OSSDefaultClause::Type::Firstprivate:
            defaultClauseOperand_ =
              mlir::oss::ClauseDefaultAttr::get(firOpBuilder.getContext(), mlir::oss::ClauseDefault::deffirstprivate);
            break;
          case Fortran::parser::OSSDefaultClause::Type::Shared:
            defaultClauseOperand_ =
              mlir::oss::ClauseDefaultAttr::get(firOpBuilder.getContext(), mlir::oss::ClauseDefault::defshared);
            break;
          case Fortran::parser::OSSDefaultClause::Type::None:
            defaultClauseOperand_ =
              mlir::oss::ClauseDefaultAttr::get(firOpBuilder.getContext(), mlir::oss::ClauseDefault::defnone);
            break;
          }
        } else if (const auto &privateClause =
                       std::get_if<Fortran::parser::OSSClause::Private>(
                           &clause.u)) {
          // const Fortran::parser::OSSObjectList &ossObjectList = privateClause->v;
          // genObjectList(ossObjectList, converter, privateClauseOperands_);
        } else if (const auto &firstprivateClause =
                       std::get_if<Fortran::parser::OSSClause::Firstprivate>(
                           &clause.u)) {
          // const Fortran::parser::OSSObjectList &ossObjectList =
          //     firstprivateClause->v;
          // genObjectList(ossObjectList, converter, firstprivateClauseOperands_);
        } else if (const auto &sharedClause =
                       std::get_if<Fortran::parser::OSSClause::Shared>(
                           &clause.u)) {
          // const Fortran::parser::OSSObjectList &ossObjectList = sharedClause->v;
          // genObjectList(ossObjectList, converter, sharedClauseOperands_);
        } else if (const auto &chunksizeClause =
                       std::get_if<Fortran::parser::OSSClause::Chunksize>(
                           &clause.u)) {
          chunksizeClauseOperand_ = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(chunksizeClause->v), stmtCtx));
        } else if (const auto &grainsizeClause =
                       std::get_if<Fortran::parser::OSSClause::Grainsize>(
                           &clause.u)) {
          grainsizeClauseOperand_ = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(grainsizeClause->v), stmtCtx));
        } else if (const auto &dependClause =
                       std::get_if<Fortran::parser::OSSClause::Depend>(
                           &clause.u)) {
          const auto &ossDependClause{dependClause->v};
          const auto &inOut{std::get<Fortran::parser::OSSDependClause::InOut>(ossDependClause.u)};
          const auto &dependType{std::get<Fortran::parser::OSSDependenceType>(inOut.t)};
          const Fortran::parser::OSSObjectList &ossObjectList{std::get<Fortran::parser::OSSObjectList>(inOut.t)};
          for (const auto &ossObject : ossObjectList.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            switch (dependType.v) {
            case Fortran::parser::OSSDependenceType::Type::In:
              inClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Out:
              outClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Inout:
              inoutClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Inoutset:
              concurrentClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Mutexinoutset:
              commutativeClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Weakin:
              weakinClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Weakout:
              weakoutClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Weakinout:
              weakinoutClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Weakinoutset:
              weakconcurrentClauseOperands_.push_back(op);
              break;
            case Fortran::parser::OSSDependenceType::Type::Weakmutexinoutset:
              weakcommutativeClauseOperands_.push_back(op);
              break;
            }
          }
        } else if (const auto &inClause =
                       std::get_if<Fortran::parser::OSSClause::In>(
                           &clause.u)) {
          for (const auto &ossObject : inClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            inClauseOperands_.push_back(op);
          }
        } else if (const auto &outClause =
                       std::get_if<Fortran::parser::OSSClause::Out>(
                           &clause.u)) {
          for (const auto &ossObject : outClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            outClauseOperands_.push_back(op);
          }
        } else if (const auto &inoutClause =
                       std::get_if<Fortran::parser::OSSClause::Inout>(
                           &clause.u)) {
          for (const auto &ossObject : inoutClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            inoutClauseOperands_.push_back(op);
          }
        } else if (const auto &concurrentClause =
                       std::get_if<Fortran::parser::OSSClause::Concurrent>(
                           &clause.u)) {
          for (const auto &ossObject : concurrentClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            concurrentClauseOperands_.push_back(op);
          }
        } else if (const auto &commutativeClause =
                       std::get_if<Fortran::parser::OSSClause::Commutative>(
                           &clause.u)) {
          for (const auto &ossObject : commutativeClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            commutativeClauseOperands_.push_back(op);
          }
        } else if (const auto &weakinClause =
                       std::get_if<Fortran::parser::OSSClause::Weakin>(
                           &clause.u)) {
          for (const auto &ossObject : weakinClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            weakinClauseOperands_.push_back(op);
          }
        } else if (const auto &weakoutClause =
                       std::get_if<Fortran::parser::OSSClause::Weakout>(
                           &clause.u)) {
          for (const auto &ossObject : weakoutClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            weakoutClauseOperands_.push_back(op);
          }
        } else if (const auto &weakinoutClause =
                       std::get_if<Fortran::parser::OSSClause::Weakinout>(
                           &clause.u)) {
          for (const auto &ossObject : weakinoutClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            weakinoutClauseOperands_.push_back(op);
          }
        } else if (const auto &weakconcurrentClause =
                       std::get_if<Fortran::parser::OSSClause::Weakconcurrent>(
                           &clause.u)) {
          for (const auto &ossObject : weakconcurrentClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            weakconcurrentClauseOperands_.push_back(op);
          }
        } else if (const auto &weakcommutativeClause =
                       std::get_if<Fortran::parser::OSSClause::Weakcommutative>(
                           &clause.u)) {
          for (const auto &ossObject : weakcommutativeClause->v.v) {
            mlir::Value op = getComputeDep(converter, context, eval, scope, stmtCtx, ossObject);
            weakcommutativeClauseOperands_.push_back(op);
          }
        }
      }
    }

    mlir::Value ifClauseOperand() const { return ifClauseOperand_; }
    mlir::Value finalClauseOperand() const { return finalClauseOperand_; }
    mlir::Value costClauseOperand() const { return costClauseOperand_; }
    mlir::Value priorityClauseOperand() const { return priorityClauseOperand_; }
    mlir::oss::ClauseDefaultAttr defaultClauseOperand() const { return defaultClauseOperand_; }
    llvm::ArrayRef<mlir::Value> privateClauseOperands() const { return privateClauseOperands_.getArrayRef(); }
    llvm::ArrayRef<mlir::Value> firstprivateClauseOperands() const { return firstprivateClauseOperands_.getArrayRef(); }
    llvm::ArrayRef<mlir::Value> copyClauseOperands() const { return copyClauseOperands_.getArrayRef(); }
    llvm::ArrayRef<mlir::Value> initClauseOperands() const { return initClauseOperands_.getArrayRef(); }
    llvm::ArrayRef<mlir::Value> deinitClauseOperands() const { return deinitClauseOperands_.getArrayRef(); }
    llvm::ArrayRef<mlir::Value> sharedClauseOperands() const { return sharedClauseOperands_.getArrayRef(); }
    llvm::ArrayRef<mlir::Value> vlaDimsOperands() const { return vlaDimsOperands_; }
    llvm::ArrayRef<mlir::Value> captureOperands() const { return capturesOperands_.getArrayRef(); }
    mlir::Value chunksizeClauseOperand() const { return chunksizeClauseOperand_; }
    mlir::Value grainsizeClauseOperand() const { return grainsizeClauseOperand_; }
    llvm::ArrayRef<mlir::Value> inClauseOperands() const { return inClauseOperands_; }
    llvm::ArrayRef<mlir::Value> outClauseOperands() const { return outClauseOperands_; }
    llvm::ArrayRef<mlir::Value> inoutClauseOperands() const { return inoutClauseOperands_; }
    llvm::ArrayRef<mlir::Value> concurrentClauseOperands() const { return concurrentClauseOperands_; }
    llvm::ArrayRef<mlir::Value> commutativeClauseOperands() const { return commutativeClauseOperands_; }
    llvm::ArrayRef<mlir::Value> weakinClauseOperands() const { return weakinClauseOperands_; }
    llvm::ArrayRef<mlir::Value> weakoutClauseOperands() const { return weakoutClauseOperands_; }
    llvm::ArrayRef<mlir::Value> weakinoutClauseOperands() const { return weakinoutClauseOperands_; }
    llvm::ArrayRef<mlir::Value> weakconcurrentClauseOperands() const { return weakconcurrentClauseOperands_; }
    llvm::ArrayRef<mlir::Value> weakcommutativeClauseOperands() const { return weakcommutativeClauseOperands_; }
    bool empty() const {
      return !ifClauseOperand_ &&
             !finalClauseOperand_ &&
             !costClauseOperand_ &&
             !priorityClauseOperand_ &&
             !defaultClauseOperand_ &&
             privateClauseOperands_.empty() &&
             firstprivateClauseOperands_.empty() &&
             copyClauseOperands_.empty() &&
             initClauseOperands_.empty() &&
             deinitClauseOperands_.empty() &&
             sharedClauseOperands_.empty() &&
             vlaDimsOperands_.empty() &&
             capturesOperands_.empty() &&
             !chunksizeClauseOperand_ &&
             !grainsizeClauseOperand_ &&
             inClauseOperands_.empty() &&
             outClauseOperands_.empty() &&
             inoutClauseOperands_.empty() &&
             concurrentClauseOperands_.empty() &&
             commutativeClauseOperands_.empty() &&
             weakinClauseOperands_.empty() &&
             weakoutClauseOperands_.empty() &&
             weakinoutClauseOperands_.empty() &&
             weakconcurrentClauseOperands_.empty() &&
             weakcommutativeClauseOperands_.empty();
    }

  private:
    Fortran::lower::AbstractConverter &converter;
    Fortran::semantics::SemanticsContext &context;
    Fortran::lower::pft::Evaluation &eval;
    const Fortran::semantics::Scope &scope;
    Fortran::lower::StatementContext &stmtCtx;
    mlir::Value ifClauseOperand_;
    mlir::Value finalClauseOperand_;
    mlir::Value costClauseOperand_;
    mlir::Value priorityClauseOperand_;
    mlir::oss::ClauseDefaultAttr defaultClauseOperand_;
    llvm::SetVector<mlir::Value> privateClauseOperands_;
    llvm::SetVector<mlir::Value> firstprivateClauseOperands_;
    // TODO: should this be a std::map or something to avoid
    // creating copy functions for symbols with the same type?
    llvm::SetVector<mlir::Value> copyClauseOperands_;
    llvm::SetVector<mlir::Value> initClauseOperands_;
    llvm::SetVector<mlir::Value> deinitClauseOperands_;
    llvm::SetVector<mlir::Value> sharedClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> vlaDimsOperands_;
    llvm::SetVector<mlir::Value> capturesOperands_;
    mlir::Value chunksizeClauseOperand_;
    mlir::Value grainsizeClauseOperand_;
    llvm::SmallVector<mlir::Value, 4> inClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> outClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> inoutClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> concurrentClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> commutativeClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> weakinClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> weakoutClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> weakinoutClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> weakconcurrentClauseOperands_;
    llvm::SmallVector<mlir::Value, 4> weakcommutativeClauseOperands_;
  };
} // namespace

static void genOSS(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &context,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OmpSsSimpleStandaloneConstruct
                       &simpleStandaloneConstruct,
                   const Fortran::lower::ImplicitDSAs &implicitDSAs) {
  const auto &directive =
      std::get<Fortran::parser::OSSSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);

  Fortran::lower::StatementContext stmtCtx;

  Fortran::lower::pft::FunctionLikeUnit *funit = eval.getOwningProcedure();
  const Fortran::semantics::Scope &scope = funit->getScope();

  const auto &clauseList =
      std::get<Fortran::parser::OSSClauseList>(simpleStandaloneConstruct.t);
  OSSClausesVisitor clausesVisitor(converter, context, eval, scope, stmtCtx, implicitDSAs);
  clausesVisitor.gatherClauseList(clauseList);

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  llvm::ArrayRef<mlir::Type> argTy;
  switch (directive.v) {
  default:
    break;
  case llvm::oss::Directive::OSSD_taskwait:
    if (clausesVisitor.empty()) {
      firOpBuilder.create<mlir::oss::TaskwaitOp>(
          currentLocation);
    } else {
      auto taskOp = firOpBuilder.create<mlir::oss::TaskOp>(
          currentLocation, argTy,
          firOpBuilder.createIntegerConstant(currentLocation, firOpBuilder.getI1Type(), 0),
          clausesVisitor.finalClauseOperand(),
          clausesVisitor.costClauseOperand(),
          clausesVisitor.priorityClauseOperand(),
          clausesVisitor.defaultClauseOperand(),
          clausesVisitor.privateClauseOperands(),
          clausesVisitor.firstprivateClauseOperands(),
          clausesVisitor.copyClauseOperands(),
          clausesVisitor.initClauseOperands(),
          clausesVisitor.deinitClauseOperands(),
          clausesVisitor.sharedClauseOperands(),
          clausesVisitor.vlaDimsOperands(),
          clausesVisitor.captureOperands(),
          clausesVisitor.inClauseOperands(),
          clausesVisitor.outClauseOperands(),
          clausesVisitor.inoutClauseOperands(),
          clausesVisitor.concurrentClauseOperands(),
          clausesVisitor.commutativeClauseOperands(),
          clausesVisitor.weakinClauseOperands(),
          clausesVisitor.weakoutClauseOperands(),
          clausesVisitor.weakinoutClauseOperands(),
          clausesVisitor.weakconcurrentClauseOperands(),
          clausesVisitor.weakcommutativeClauseOperands());
      firOpBuilder.createBlock(&taskOp.getRegion());
      auto &block = taskOp.getRegion().back();
      firOpBuilder.setInsertionPointToStart(&block);
      // Ensure the block is well-formed.
      firOpBuilder.create<mlir::oss::TerminatorOp>(currentLocation);
      // Reset the insertion point to the start of the first block.
      firOpBuilder.setInsertionPointToStart(&block);
    }
    break;
  case llvm::oss::Directive::OSSD_release:
    firOpBuilder.create<mlir::oss::ReleaseOp>(
        currentLocation, argTy,
        clausesVisitor.inClauseOperands(),
        clausesVisitor.outClauseOperands(),
        clausesVisitor.inoutClauseOperands(),
        clausesVisitor.weakinClauseOperands(),
        clausesVisitor.weakoutClauseOperands(),
        clausesVisitor.weakinoutClauseOperands());
    break;
  }
}

static void
genOSS(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &context,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OmpSsStandaloneConstruct &standaloneConstruct,
       const Fortran::lower::ImplicitDSAs &implicitDSAs) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OmpSsSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOSS(converter, context, eval, simpleStandaloneConstruct, implicitDSAs);
          },
      },
      standaloneConstruct.u);
}

static void
genOSS(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &context,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OmpSsBlockConstruct &blockConstruct,
       const Fortran::lower::ImplicitDSAs &implicitDSAs) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OSSBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::OSSBlockDirective>(beginBlockDirective.t);

  Fortran::lower::pft::FunctionLikeUnit *funit = eval.getOwningProcedure();
  const Fortran::semantics::Scope &scope = funit->getScope();

  Fortran::lower::StatementContext stmtCtx;

  const auto &clauseList =
      std::get<Fortran::parser::OSSClauseList>(beginBlockDirective.t);
  OSSClausesVisitor clausesVisitor(converter, context, eval, scope, stmtCtx, implicitDSAs);
  clausesVisitor.gatherClauseList(clauseList);

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  llvm::ArrayRef<mlir::Type> argTy;
  if (blockDirective.v == llvm::oss::OSSD_task) {
    // Create and insert the operation.
    auto taskOp = firOpBuilder.create<mlir::oss::TaskOp>(
        currentLocation, argTy,
        clausesVisitor.ifClauseOperand(),
        clausesVisitor.finalClauseOperand(),
        clausesVisitor.costClauseOperand(),
        clausesVisitor.priorityClauseOperand(),
        clausesVisitor.defaultClauseOperand(),
        clausesVisitor.privateClauseOperands(),
        clausesVisitor.firstprivateClauseOperands(),
        clausesVisitor.copyClauseOperands(),
        clausesVisitor.initClauseOperands(),
        clausesVisitor.deinitClauseOperands(),
        clausesVisitor.sharedClauseOperands(),
        clausesVisitor.vlaDimsOperands(),
        clausesVisitor.captureOperands(),
        clausesVisitor.inClauseOperands(),
        clausesVisitor.outClauseOperands(),
        clausesVisitor.inoutClauseOperands(),
        clausesVisitor.concurrentClauseOperands(),
        clausesVisitor.commutativeClauseOperands(),
        clausesVisitor.weakinClauseOperands(),
        clausesVisitor.weakoutClauseOperands(),
        clausesVisitor.weakinoutClauseOperands(),
        clausesVisitor.weakconcurrentClauseOperands(),
        clausesVisitor.weakcommutativeClauseOperands());
    createBodyOfOpWithPreface<mlir::oss::TaskOp>(taskOp, converter, firOpBuilder, currentLocation, eval, implicitDSAs);
  }
}

static void
genOSS(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &context,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OmpSsLoopConstruct &loopConstruct,
       const Fortran::lower::ImplicitDSAs &implicitDSAs) {
  const auto &beginLoopDir{std::get<Fortran::parser::OSSBeginLoopDirective>(loopConstruct.t)};
  const auto &loopDir{std::get<Fortran::parser::OSSLoopDirective>(beginLoopDir.t)};

  Fortran::lower::pft::FunctionLikeUnit *funit = eval.getOwningProcedure();
  const Fortran::semantics::Scope &scope = funit->getScope();

  Fortran::lower::StatementContext stmtCtx;

  const auto &clauseList =
      std::get<Fortran::parser::OSSClauseList>(beginLoopDir.t);
  OSSClausesVisitor clausesVisitor(converter, context, eval, scope, stmtCtx, implicitDSAs);
  clausesVisitor.gatherClauseList(clauseList);

  // NOTE: do this after visiting clauses to bind symbols
  OSSLoopBoundsVisitor loopBoundsVisitor(converter, stmtCtx);
  loopBoundsVisitor.Walk(loopConstruct);

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  llvm::ArrayRef<mlir::Type> argTy;

  // Create and insert the operation.
  if (loopDir.v == llvm::oss::OSSD_taskloop) {
    auto taskloopOp = firOpBuilder.create<mlir::oss::TaskloopOp>(
        currentLocation, argTy,
        loopBoundsVisitor.lowerBoundOperand(),
        loopBoundsVisitor.upperBoundOperand(),
        loopBoundsVisitor.stepOperand(),
        loopBoundsVisitor.loopTypeOperand(),
        loopBoundsVisitor.indVarOperand(),
        clausesVisitor.ifClauseOperand(),
        clausesVisitor.finalClauseOperand(),
        clausesVisitor.costClauseOperand(),
        clausesVisitor.priorityClauseOperand(),
        clausesVisitor.defaultClauseOperand(),
        clausesVisitor.privateClauseOperands(),
        clausesVisitor.firstprivateClauseOperands(),
        clausesVisitor.copyClauseOperands(),
        clausesVisitor.initClauseOperands(),
        clausesVisitor.deinitClauseOperands(),
        clausesVisitor.sharedClauseOperands(),
        clausesVisitor.vlaDimsOperands(),
        clausesVisitor.captureOperands(),
        clausesVisitor.inClauseOperands(),
        clausesVisitor.outClauseOperands(),
        clausesVisitor.inoutClauseOperands(),
        clausesVisitor.concurrentClauseOperands(),
        clausesVisitor.commutativeClauseOperands(),
        clausesVisitor.weakinClauseOperands(),
        clausesVisitor.weakoutClauseOperands(),
        clausesVisitor.weakinoutClauseOperands(),
        clausesVisitor.weakconcurrentClauseOperands(),
        clausesVisitor.weakcommutativeClauseOperands(),
        clausesVisitor.grainsizeClauseOperand());
    createBodyOfOp<mlir::oss::TaskloopOp>(taskloopOp, firOpBuilder, currentLocation, eval);
  } else if (loopDir.v == llvm::oss::OSSD_task_for) {
    auto taskForOp = firOpBuilder.create<mlir::oss::TaskForOp>(
        currentLocation, argTy,
        loopBoundsVisitor.lowerBoundOperand(),
        loopBoundsVisitor.upperBoundOperand(),
        loopBoundsVisitor.stepOperand(),
        loopBoundsVisitor.loopTypeOperand(),
        loopBoundsVisitor.indVarOperand(),
        clausesVisitor.ifClauseOperand(),
        clausesVisitor.finalClauseOperand(),
        clausesVisitor.costClauseOperand(),
        clausesVisitor.priorityClauseOperand(),
        clausesVisitor.defaultClauseOperand(),
        clausesVisitor.privateClauseOperands(),
        clausesVisitor.firstprivateClauseOperands(),
        clausesVisitor.copyClauseOperands(),
        clausesVisitor.initClauseOperands(),
        clausesVisitor.deinitClauseOperands(),
        clausesVisitor.sharedClauseOperands(),
        clausesVisitor.vlaDimsOperands(),
        clausesVisitor.captureOperands(),
        clausesVisitor.inClauseOperands(),
        clausesVisitor.outClauseOperands(),
        clausesVisitor.inoutClauseOperands(),
        clausesVisitor.concurrentClauseOperands(),
        clausesVisitor.commutativeClauseOperands(),
        clausesVisitor.weakinClauseOperands(),
        clausesVisitor.weakoutClauseOperands(),
        clausesVisitor.weakinoutClauseOperands(),
        clausesVisitor.weakconcurrentClauseOperands(),
        clausesVisitor.weakcommutativeClauseOperands(),
        clausesVisitor.chunksizeClauseOperand());
    createBodyOfOp<mlir::oss::TaskForOp>(taskForOp, firOpBuilder, currentLocation, eval);
  } else if (loopDir.v == llvm::oss::OSSD_taskloop_for) {
    auto taskloopForOp = firOpBuilder.create<mlir::oss::TaskloopForOp>(
        currentLocation, argTy,
        loopBoundsVisitor.lowerBoundOperand(),
        loopBoundsVisitor.upperBoundOperand(),
        loopBoundsVisitor.stepOperand(),
        loopBoundsVisitor.loopTypeOperand(),
        loopBoundsVisitor.indVarOperand(),
        clausesVisitor.ifClauseOperand(),
        clausesVisitor.finalClauseOperand(),
        clausesVisitor.costClauseOperand(),
        clausesVisitor.priorityClauseOperand(),
        clausesVisitor.defaultClauseOperand(),
        clausesVisitor.privateClauseOperands(),
        clausesVisitor.firstprivateClauseOperands(),
        clausesVisitor.copyClauseOperands(),
        clausesVisitor.initClauseOperands(),
        clausesVisitor.deinitClauseOperands(),
        clausesVisitor.sharedClauseOperands(),
        clausesVisitor.vlaDimsOperands(),
        clausesVisitor.captureOperands(),
        clausesVisitor.inClauseOperands(),
        clausesVisitor.outClauseOperands(),
        clausesVisitor.inoutClauseOperands(),
        clausesVisitor.concurrentClauseOperands(),
        clausesVisitor.commutativeClauseOperands(),
        clausesVisitor.weakinClauseOperands(),
        clausesVisitor.weakoutClauseOperands(),
        clausesVisitor.weakinoutClauseOperands(),
        clausesVisitor.weakconcurrentClauseOperands(),
        clausesVisitor.weakcommutativeClauseOperands(),
        clausesVisitor.chunksizeClauseOperand(),
        clausesVisitor.grainsizeClauseOperand());
    createBodyOfOp<mlir::oss::TaskloopForOp>(taskloopForOp, firOpBuilder, currentLocation, eval);
  } else {
    llvm_unreachable("Unexpected loop directive kind");
  }
}

void Fortran::lower::genOmpSsConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OmpSsConstruct &ossConstruct,
    const Fortran::lower::ImplicitDSAs &implicitDSAs,
    Fortran::semantics::SemanticsContext &context) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OmpSsStandaloneConstruct
                  &standaloneConstruct) {
            genOSS(converter, context, eval, standaloneConstruct, implicitDSAs);
          },
          [&](const Fortran::parser::OmpSsLoopConstruct &loopConstruct) {
            genOSS(converter, context, eval, loopConstruct, implicitDSAs);
          },
          [&](const Fortran::parser::OmpSsBlockConstruct &blockConstruct) {
            genOSS(converter, context, eval, blockConstruct, implicitDSAs);
          },
      },
      ossConstruct.u);
}

mlir::Value Fortran::lower::genOmpSsTaskSubroutine(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::semantics::Scope &scope,
    const Fortran::parser::OmpSsOutlineTaskConstruct &outlineTask,
    Fortran::semantics::SemanticsContext &context,
    const Fortran::evaluate::ProcedureRef &procRef,
    StatementContext &stmtCtx) {

  const auto &simpleConstruct{std::get<parser::OmpSsSimpleOutlineTaskConstruct>(outlineTask.t)};
  const auto &clauseList{std::get<parser::OSSClauseList>(simpleConstruct.t)};
  
  Fortran::lower::CallerInterface caller(procRef, converter);

  Fortran::lower::SymMap localSymbols;

  // Create allocas to store argument expressions
  Fortran::lower::createOSSAllocasForArgs(
    localSymbols, converter, caller, stmtCtx);

  converter.getLocalSymbols().symbolMapStack.emplace_back(localSymbols.symbolMapStack.back());

  OSSClausesVisitor clausesVisitor(converter, context, eval, scope, stmtCtx);
  clausesVisitor.gatherClauseList(clauseList);
  
  auto loc = converter.getCurrentLocation();
  auto &firOpBuilder = converter.getFirOpBuilder();
  llvm::ArrayRef<mlir::Type> argTy;
  llvm::SmallVector<mlir::Value> val_private; /*No needed*/
  llvm::SmallVector<mlir::Value> val_firstprivate; /*No needed*/
  // TODO: is this needed?
  llvm::SmallVector<mlir::Value> val_copy; /*No needed*/
  llvm::SmallVector<mlir::Value> val_init; /*No needed*/
  llvm::SmallVector<mlir::Value> val_deinit; /*No needed*/
  llvm::SmallVector<mlir::Value> val_shared = 
        Fortran::lower::fillDSAs(caller, converter, stmtCtx);

  converter.getLocalSymbols().popScope();

  auto taskOp = firOpBuilder.create<mlir::oss::TaskOp>(
    loc, argTy,
    clausesVisitor.ifClauseOperand(),
    clausesVisitor.finalClauseOperand(),
    clausesVisitor.costClauseOperand(),
    clausesVisitor.priorityClauseOperand(),
    clausesVisitor.defaultClauseOperand(),
    val_private,
    val_firstprivate,
    val_copy,
    val_init,
    val_deinit,
    val_shared,
    clausesVisitor.vlaDimsOperands(),
    clausesVisitor.captureOperands(),
    clausesVisitor.inClauseOperands(),
    clausesVisitor.outClauseOperands(),
    clausesVisitor.inoutClauseOperands(),
    clausesVisitor.concurrentClauseOperands(),
    clausesVisitor.commutativeClauseOperands(),
    clausesVisitor.weakinClauseOperands(),
    clausesVisitor.weakoutClauseOperands(),
    clausesVisitor.weakinoutClauseOperands(),
    clausesVisitor.weakconcurrentClauseOperands(),
    clausesVisitor.weakcommutativeClauseOperands());

  firOpBuilder.createBlock(&taskOp.getRegion());
  auto &block = taskOp.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);
  // Ensure the block is well-formed.
  firOpBuilder.create<mlir::oss::TerminatorOp>(loc);
  // Reset the insertion point to the start of the first block.
  firOpBuilder.setInsertionPointToStart(&block);

  return fir::getBase(Fortran::lower::createOSSCallOpWithAllocas(localSymbols, converter, caller, stmtCtx));
}

