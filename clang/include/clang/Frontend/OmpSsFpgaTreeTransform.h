//===- OmpssFpgaTreeTransform.h - Transformations to the AST ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the AST transformations necessary for producing a valid
/// wrapper.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_OMPSSFPGATREETRANSFORM_H
#define LLVM_CLANG_FRONTEND_OMPSSFPGATREETRANSFORM_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/APSInt.h"
namespace clang {

static constexpr uint8_t InstrumentationEventBurstBegin = 0b0000'0000;
static constexpr uint8_t InstrumentationEventBurstEnd = 0b1000'0000;

#define BURST(name, val)                                                       \
  name##Begin = InstrumentationEventBurstBegin | val,                          \
  name##End = InstrumentationEventBurstEnd | val

enum class FPGAInstrumentationEvents : uint8_t {
  BURST(APICall, 85),
  BURST(DevCopyIn, 78),
  BURST(DevCopyOut, 79),
  BURST(DevExec, 80),
};
#undef BURST

enum class FPGAInstrumentationApiCalls : uint64_t {
  WaitTasks = 1,
  SetLock = 2,
  UnsetLock = 3,
  CreateTask = 5
};

enum class WrapperPort {
  OMPIF_RANK = 0,
  OMPIF_SIZE = 1,
  MEMORY_PORT = 2,
  INPORT = 3,
  OUTPORT = 4,
  SPAWN_INPORT = 5,
  NUM_PORTS = 6
};

using WrapperPortMap =
    llvm::SmallMapVector<const Decl *,
                         std::array<bool, size_t(WrapperPort::NUM_PORTS)>, 16>;

struct LocalmemInfo {
  int ParamIdx = -1;
  const OSSArrayShapingExpr *FixedArrayRef;
  enum Dir { IN = 0b01, OUT = 0b10, INOUT = 0b11 };
  Dir dir;
};

class FPGAFunctionTreeVisitor
    : public RecursiveASTVisitor<FPGAFunctionTreeVisitor> {

  struct FunctionCallTree {
    const Decl *symbol;
    FunctionCallTree *parent;

    FunctionCallTree(const Decl *symbol, FunctionCallTree *parent)
        : symbol(symbol), parent(parent) {}
  };

  FunctionCallTree Top;
  FunctionCallTree *Current;
  WrapperPortMap &wrapperPortMap;

  void propagatePort(WrapperPort port);

public:
  bool CreatesTasks = false;
  bool UsesOmpif = false;
  bool MemcpyWideport = false;
  bool UsesLock = false;

  FPGAFunctionTreeVisitor(FunctionDecl *startSymbol,
                          WrapperPortMap &wrapperPortMap);

  bool VisitOSSTaskDirective(OSSTaskDirective *);
  bool VisitOSSTaskwaitDirective(OSSTaskwaitDirective *);
  bool VisitCXXConstructExpr(CXXConstructExpr *n);
  bool VisitOSSCriticalDirective(OSSCriticalDirective *n);
  bool VisitCallExpr(CallExpr *n);
};

std::pair<bool, ReplacementMap>
OmpssFpgaTreeTransform(clang::ASTContext &Ctx,
                       clang::IdentifierTable &identifierTable,
                       WrapperPortMap &WrapperPortMap, uint64_t FpgaPortWidth,
                       bool CreatesTasks, bool instrumented);

using ParamDependencyMap =
    llvm::SmallDenseMap<const ParmVarDecl *,
                        std::pair<const Expr *, LocalmemInfo::Dir>>;
// Compute the direction tags of the parameters. Do note that not
// all parameters are guaranteed to be present
ParamDependencyMap computeDependencyMap(OSSTaskDeclAttr *taskAttr,
                                        bool includeNonArrays = false);

llvm::SmallDenseMap<const clang::ParmVarDecl *, LocalmemInfo>
ComputeLocalmems(FunctionDecl *FD);

QualType DerefOnceTypePointerTo(QualType type);
QualType GetElementTypePointerTo(QualType type);
QualType LocalmemArrayType(ASTContext &Ctx,
                           const OSSArrayShapingExpr *arrayType);
uint64_t ComputeArrayRefSize(ASTContext &Ctx,
                             const OSSArrayShapingExpr *arrayType,
                             uint64_t baseType = 1);

std::optional<llvm::APSInt> extractIntegerConstantFromExpr(ASTContext &Ctx,
                                                           const Expr *expr);

uint64_t GenOnto(ASTContext &Ctx, FunctionDecl *FD);
} // namespace clang

#endif