//===- OmpssFpgaTreeTransform.cpp - Transformations to the AST ------------===//
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

#include "clang/Frontend/OmpSsFpgaTreeTransform.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTFwd.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>

using namespace clang;
namespace {
class OmpSsFpgaTreeTransformVisitor
    : public RecursiveASTVisitor<OmpSsFpgaTreeTransformVisitor> {

  using Inherited = RecursiveASTVisitor<OmpSsFpgaTreeTransformVisitor>;
  ASTContext &Ctx;
  IdentifierTable &IdentifierTable;
  WrapperPortMap &WrapperPortMap;
  PrintingPolicy PrintPol;

  uint64_t FpgaPortWidth;
  bool CreatesTasks;

  QualType OmpIfRankType;
  IdentifierInfo *OmpIfRankIdentifier;
  ParmVarDecl *OmpIfRank;
  QualType OmpIfSizeType;
  IdentifierInfo *OmpIfSizeIdentifier;
  ParmVarDecl *OmpIfSize;
  QualType SpawnInPortType;
  IdentifierInfo *SpawnInPortIdentifier;
  ParmVarDecl *SpawnInPort;
  QualType InPortType;
  IdentifierInfo *InPortIdentifier;
  ParmVarDecl *InPort;
  QualType OutPortType;
  IdentifierInfo *OutPortIdentifier;
  ParmVarDecl *OutPort;
  QualType MemPortType;
  IdentifierInfo *MemPortIdentifier;
  ParmVarDecl *MemPort;

  QualType McxxSetLockType;
  IdentifierInfo *McxxSetLockIdentifier;
  FunctionDecl *McxxSetLock;

  QualType McxxUnsetLockType;
  IdentifierInfo *McxxUnsetLockIdentifier;
  FunctionDecl *McxxUnsetLock;

  QualType McxxTaskwaitType;
  IdentifierInfo *McxxTaskwaitIdentifier;
  FunctionDecl *McxxTaskwait;

  template <class T> struct ReplacementBlock {
    T *original;
    T *replaced;
  };
  llvm::SmallVector<ReplacementBlock<Stmt>, 2> replacementStmt;
  llvm::SmallVector<ReplacementBlock<Expr>, 2> replacementExpr;

  template <class T>
  void addReplacementOpStmt(T *OriginalPos, Stmt *Replacement) {
    static_assert(sizeof(OSSRedirectStmt) <= sizeof(T),
                  "expected replaced node to be smaller or equal than the "
                  "redirection node");
    replacementStmt.push_back(ReplacementBlock<Stmt>{OriginalPos, Replacement});
  }

  template <class T>
  void addReplacementOpExpr(T *OriginalPos, Expr *Replacement) {
    static_assert(sizeof(OSSRedirectExpr) <= sizeof(T),
                  "expected replaced node to be smaller or equal than the "
                  "redirection node");
    replacementExpr.push_back(ReplacementBlock<Expr>{OriginalPos, Replacement});
  }

  DeclRefExpr *makeDeclRefExpr(ValueDecl *Decl) const {
    assert(Decl && "Decl must not be null");
    return DeclRefExpr::Create(Ctx, NestedNameSpecifierLoc{}, SourceLocation{},
                               Decl, false, SourceLocation{},
                               Decl->getType().getNonReferenceType(), {});
  }

  template <typename Num> IntegerLiteral *makeIntegerLiteral(Num value) const {
    static_assert(std::is_integral_v<Num>, "Use a number as a parameter");
    return IntegerLiteral::Create(
        Ctx, llvm::APInt(sizeof(value) * CHAR_BIT, value, value < 0),
        Ctx.getIntTypeForBitwidth(sizeof(value) * CHAR_BIT, value < 0),
        SourceLocation{});
  }

  StringRef AllocatedStringRef(StringRef original) {
    char *mem = reinterpret_cast<char *>(
        Ctx.Allocate(original.size() + 1, sizeof(void *)));
    memcpy(mem, original.data(), original.size());
    mem[original.size()] = '\0';
    return StringRef(mem);
  }

  // WARNING: If the function decl is going to be
  // printed, it's more performant to directly use the types, decls
  // and so on from the original function.
  CallExpr *makeCallToWithDifferentParams(CallExpr *original,
                                          llvm::ArrayRef<Expr *> parameters) {
    llvm::SmallVector<QualType, 4> paramTypes(parameters.size());
    int i = 0;
    for (auto &&param : parameters) {
      paramTypes[i] = param->getType();
      ++i;
    }

    auto FunctionType =
        Ctx.getFunctionType(original->getCallReturnType(Ctx), paramTypes,
                            FunctionProtoType::ExtProtoInfo());

    auto *FunctionIdentifier =
        llvm::dyn_cast<FunctionDecl>(original->getCalleeDecl())
            ->getIdentifier();
    auto *FunctionDecl = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl()->getDeclContext(), {}, {},
        DeclarationName(FunctionIdentifier), FunctionType, nullptr, SC_None);

    makeDeclRefExpr(McxxSetLock);
    return CallExpr::Create(Ctx, makeDeclRefExpr(FunctionDecl), parameters,
                            original->getType(), original->getValueKind(),
                            SourceLocation{}, original->getFPFeatures());
  }

public:
  OmpSsFpgaTreeTransformVisitor(ASTContext &Ctx,
                                clang::IdentifierTable &IdentifierTable,
                                ::WrapperPortMap &WrapperPortMap,
                                uint64_t FpgaPortWidth, bool CreatesTasks)
      : Inherited(), Ctx(Ctx), IdentifierTable(IdentifierTable),
        WrapperPortMap(WrapperPortMap), PrintPol(Ctx.getLangOpts()),
        FpgaPortWidth(FpgaPortWidth), CreatesTasks(CreatesTasks) {

    OmpIfRankType = Ctx.UnsignedCharTy;
    OmpIfRankIdentifier = &IdentifierTable.get("__ompif_rank");

    OmpIfSizeType = Ctx.UnsignedCharTy;
    OmpIfSizeIdentifier = &IdentifierTable.get("__ompif_size");

    SpawnInPortType = Ctx.getLValueReferenceType(
        Ctx.getPrintableASTType("hls::stream<ap_uint<8> >"));
    SpawnInPortIdentifier = &IdentifierTable.get("mcxx_spawnInPort");

    InPortType = Ctx.getLValueReferenceType(
        Ctx.getPrintableASTType("hls::stream<ap_uint<64> >"));
    InPortIdentifier = &IdentifierTable.get("mcxx_inPort");

    OutPortType = Ctx.getLValueReferenceType(
        Ctx.getPrintableASTType("hls::stream<mcxx_outaxis>"));
    OutPortIdentifier = &IdentifierTable.get("mcxx_outPort");

    MemPortType = Ctx.getPointerType(Ctx.getPrintableASTType(
        "ap_uint<" + std::to_string(FpgaPortWidth) + ">"));
    MemPortIdentifier = &IdentifierTable.get("mcxx_memport");

    McxxSetLockType = Ctx.getFunctionType(Ctx.VoidTy, {InPortType, OutPortType},
                                          FunctionProtoType::ExtProtoInfo());
    McxxSetLockIdentifier = &IdentifierTable.get("mcxx_set_lock");
    McxxSetLock = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl()->getDeclContext(), {}, {},
        DeclarationName(McxxSetLockIdentifier), McxxSetLockType, nullptr,
        SC_None);

    McxxUnsetLockType = Ctx.getFunctionType(Ctx.VoidTy, {OutPortType},
                                            FunctionProtoType::ExtProtoInfo());
    McxxUnsetLockIdentifier = &IdentifierTable.get("mcxx_unset_lock");
    McxxUnsetLock = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl()->getDeclContext(), {}, {},
        DeclarationName(McxxUnsetLockIdentifier), McxxUnsetLockType, nullptr,
        SC_None);

    McxxTaskwaitType =
        Ctx.getFunctionType(Ctx.VoidTy, {SpawnInPortType, OutPortType},
                            FunctionProtoType::ExtProtoInfo());
    McxxTaskwaitIdentifier = &IdentifierTable.get("mcxx_taskwait");
    McxxTaskwait = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl()->getDeclContext(), {}, {},
        DeclarationName(McxxTaskwaitIdentifier), McxxTaskwaitType, nullptr,
        SC_None);
  }

  void performReplacements() {
    for (auto &&[original, replaced] : replacementStmt)
      new (reinterpret_cast<void *>(original)) OSSRedirectStmt(replaced);
    for (auto &&[original, replaced] : replacementExpr)
      new (reinterpret_cast<void *>(original)) OSSRedirectExpr(replaced);
  }

  bool VisitVarDecl(VarDecl *decl) {
    QualType type = decl->getType();
    if (type->isPointerType()) {
      type = type->getPointeeType().IgnoreParens();
      std::string outType;
      llvm::raw_string_ostream out(outType);
      type.print(out, PrintPol);
      type = Ctx.getPrintableASTType(
          AllocatedStringRef("__mcxx_ptr_t<" + outType + " >"));
      decl->setType(type);
      decl->setTypeSourceInfo(nullptr);
    }
    return true;
  }

  bool VisitCallExpr(CallExpr *callExpr) {
    if (callExpr->getCalleeDecl()
            ->getAttr<OSSTaskDeclAttr>()) { // TODO: Transform Task calls as
                                            // well
      return true;
    }
    // ExhaustiveVisitor<void>::visit(n);
    // Symbol sym = n.get_called().get_symbol();
    const StringRef funcName =
        dyn_cast<FunctionDecl>(callExpr->getCalleeDecl())->getName();

    if (funcName == "OMPIF_Comm_rank") {
      addReplacementOpExpr(
          callExpr,
          BinaryOperator::Create(
              Ctx, makeDeclRefExpr(OmpIfRank),
              UnaryOperator::Create(
                  Ctx, makeIntegerLiteral(1), UnaryOperatorKind::UO_Minus,
                  Ctx.IntTy, ExprValueKind::VK_PRValue,
                  ExprObjectKind::OK_Ordinary, SourceLocation{}, false, {}),
              BinaryOperatorKind::BO_Add, Ctx.IntTy, callExpr->getValueKind(),
              callExpr->getObjectKind(), SourceLocation{}, {}));
      return true;
    }
    if (funcName == "OMPIF_Comm_size") {
      addReplacementOpExpr(callExpr, makeDeclRefExpr(OmpIfRank));
      return true;
    }
    if (funcName == "OMPIF_Send" || funcName == "OMPIF_Recv") {
      llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
      arguments.reserve(callExpr->getNumArgs() + 3);
      arguments.push_back(makeIntegerLiteral(0));
      arguments.push_back(makeIntegerLiteral(0));
      arguments.push_back(makeDeclRefExpr(OutPort));

      addReplacementOpExpr(callExpr,
                           makeCallToWithDifferentParams(callExpr, arguments));
      return true;
    }
    if (funcName == "OMPIF_Allgather") {
      llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
      arguments.reserve(callExpr->getNumArgs() + 3);
      arguments.push_back(makeDeclRefExpr(OmpIfRank));
      arguments.push_back(makeDeclRefExpr(SpawnInPort));
      arguments.push_back(makeDeclRefExpr(OutPort));

      addReplacementOpExpr(callExpr,
                           makeCallToWithDifferentParams(callExpr, arguments));
      return true;
    }
    if (funcName.find("nanos6_fpga_memcpy_wideport_") == 0) {
      llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
      arguments.push_back(makeDeclRefExpr(MemPort));
      addReplacementOpExpr(callExpr,
                           makeCallToWithDifferentParams(callExpr, arguments));
      return true;
    }

    llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
    auto mapping = WrapperPortMap[callExpr->getCalleeDecl()];

    if (mapping[(int)WrapperPort::OMPIF_RANK])
      arguments.push_back(makeDeclRefExpr(OmpIfRank));
    if (mapping[(int)WrapperPort::OMPIF_SIZE])
      arguments.push_back(makeDeclRefExpr(OmpIfSize));
    if (mapping[(int)WrapperPort::SPAWN_INPORT])
      arguments.push_back(makeDeclRefExpr(SpawnInPort));
    if (mapping[(int)WrapperPort::INPORT])
      arguments.push_back(makeDeclRefExpr(InPort));
    if (mapping[(int)WrapperPort::OUTPORT])
      arguments.push_back(makeDeclRefExpr(OutPort));
    if (mapping[(int)WrapperPort::MEMORY_PORT])
      arguments.push_back(makeDeclRefExpr(MemPort));

    addReplacementOpExpr(callExpr,
                         makeCallToWithDifferentParams(callExpr, arguments));

    return true;
  }

  bool TraverseFunctionDecl(FunctionDecl *D) {
    if (D->hasAttr<OSSTaskDeclAttr>()) {
      return true;
    }
    return Inherited::TraverseFunctionDecl(D);
  }

  bool VisitFunctionDecl(FunctionDecl *funcDecl) {
    if (funcDecl->hasAttr<OSSTaskDeclAttr>()) {
      return true;
    }
    llvm::SmallVector<ParmVarDecl *, 4> NewParamInfo(funcDecl->parameters());

    auto addNewParamInfo = [&](auto &&Identifier, auto &&Type) {
      auto param = ParmVarDecl::Create(
          Ctx, funcDecl->getDeclContext(), SourceLocation{}, SourceLocation{},
          Identifier, Type, nullptr, SC_None, nullptr);
      NewParamInfo.push_back(param);
      return param;
    };

    OmpIfRank = (WrapperPortMap[funcDecl][(int)WrapperPort::OMPIF_RANK])
                    ? addNewParamInfo(OmpIfRankIdentifier, OmpIfRankType)
                    : nullptr;
    OmpIfSize = (WrapperPortMap[funcDecl][(int)WrapperPort::OMPIF_SIZE])
                    ? addNewParamInfo(OmpIfSizeIdentifier, OmpIfSizeType)
                    : nullptr;
    SpawnInPort = (WrapperPortMap[funcDecl][(int)WrapperPort::SPAWN_INPORT])
                      ? addNewParamInfo(SpawnInPortIdentifier, SpawnInPortType)
                      : nullptr;
    InPort = (WrapperPortMap[funcDecl][(int)WrapperPort::INPORT])
                 ? addNewParamInfo(InPortIdentifier, InPortType)
                 : nullptr;
    OutPort = (WrapperPortMap[funcDecl][(int)WrapperPort::OUTPORT])
                  ? addNewParamInfo(OutPortIdentifier, OutPortType)
                  : nullptr;
    MemPort = (WrapperPortMap[funcDecl][(int)WrapperPort::MEMORY_PORT])
                  ? addNewParamInfo(MemPortIdentifier, MemPortType)
                  : nullptr;

    const auto *origType = funcDecl->getType()->getAs<FunctionType>();
    llvm::SmallVector<QualType, 4> typesParam;
    typesParam.reserve(NewParamInfo.size());
    for (auto *param : NewParamInfo) {
      typesParam.push_back(param->getType());
    }
    funcDecl->setType(Ctx.getFunctionType(origType->getReturnType(), typesParam,
                                          FunctionProtoType::ExtProtoInfo()));
    funcDecl->resetParams();
    funcDecl->setParams(NewParamInfo);
    return true;
  }

  bool VisitOSSCriticalDirective(OSSCriticalDirective *n) {
    llvm::SmallVector<Stmt *, 4> stmts;

    DeclRefExpr *in = makeDeclRefExpr(InPort), *out = makeDeclRefExpr(OutPort);
    stmts.push_back(CallExpr::Create(
        Ctx, makeDeclRefExpr(McxxSetLock), {in, out},
        dyn_cast<FunctionType>(McxxSetLockType.getTypePtr())->getReturnType(),
        {}, {}, {}));

    if (n->hasAssociatedStmt())
      stmts.push_back(n->getAssociatedStmt());

    stmts.push_back(CallExpr::Create(
        Ctx, makeDeclRefExpr(McxxUnsetLock), {out},
        dyn_cast<FunctionType>(McxxUnsetLockType.getTypePtr())->getReturnType(),
        {}, {}, {}));

    auto *stmt = CompoundStmt::Create(Ctx, stmts, FPOptionsOverride(),
                                      SourceLocation{}, SourceLocation{});
    addReplacementOpStmt(n, stmt);
    return true;
  }

  bool VisitOSSTaskwaitDirective(OSSTaskwaitDirective *n) {
    DeclRefExpr *SapawnIn = makeDeclRefExpr(SpawnInPort),
                *out = makeDeclRefExpr(OutPort);

    addReplacementOpStmt(
        n,
        CallExpr::Create(Ctx, makeDeclRefExpr(McxxTaskwait), {SapawnIn, out},
                         dyn_cast<FunctionType>(McxxUnsetLockType.getTypePtr())
                             ->getReturnType(),
                         {}, {}, {}));
    return true;
  }
};
} // namespace
namespace clang {
void OmpssFpgaTreeTransform(clang::ASTContext &Ctx,
                            clang::IdentifierTable &identifierTable,
                            WrapperPortMap &WrapperPortMap,
                            uint64_t FpgaPortWidth, bool CreatesTasks) {
  OmpSsFpgaTreeTransformVisitor t(Ctx, identifierTable, WrapperPortMap,
                                  FpgaPortWidth, CreatesTasks);
  t.TraverseAST(Ctx);
  t.performReplacements();
}
} // namespace clang

void FPGAFunctionTreeVisitor::propagatePort(WrapperPort port) {
  const FunctionCallTree *node = Current;
  do {
    wrapperPortMap[node->symbol][(int)port] = true;
    node = node->parent;
  } while (node != nullptr);
}

FPGAFunctionTreeVisitor::FPGAFunctionTreeVisitor(FunctionDecl *startSymbol,
                                                 WrapperPortMap &wrapperPortMap)
    : Top(startSymbol, nullptr), Current(&Top), wrapperPortMap(wrapperPortMap) {
}

bool FPGAFunctionTreeVisitor::VisitOSSTaskDirective(OSSTaskDirective *) {
  CreatesTasks = true;
  propagatePort(WrapperPort::OUTPORT);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitOSSTaskwaitDirective(
    OSSTaskwaitDirective *) {
  CreatesTasks = true;
  propagatePort(WrapperPort::OUTPORT);
  propagatePort(WrapperPort::SPAWN_INPORT);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitCXXConstructExpr(CXXConstructExpr *n) {
  auto *body = n->getConstructor()->getBody();
  getDerived().TraverseStmt(body);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitOSSCriticalDirective(
    OSSCriticalDirective *n) {
  UsesLock = true;
  propagatePort(WrapperPort::INPORT);
  propagatePort(WrapperPort::OUTPORT);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitCallExpr(CallExpr *n) {
  auto *sym = n->getCalleeDecl();
  if (const NamedDecl *symNamed = dyn_cast<NamedDecl>(sym); symNamed) {
    auto symName = symNamed->getName();

    if (symName == "OMPIF_Comm_rank") {
      UsesOmpif = true;
      propagatePort(WrapperPort::OMPIF_RANK);
      return true;
    }
    if (symName == "OMPIF_Comm_size") {
      UsesOmpif = true;
      propagatePort(WrapperPort::OMPIF_SIZE);
      return true;
    }
    if (symName.startswith("OMPIF_")) {
      UsesOmpif = CreatesTasks = true;
      propagatePort(WrapperPort::OUTPORT);
      if (symName == "OMPIF_Allgather") {
        propagatePort(WrapperPort::OMPIF_RANK);
        propagatePort(WrapperPort::SPAWN_INPORT);
      }
    } else if (symName == "nanos6_fpga_memcpy_wideport_in" ||
               symName == "nanos6_fpga_memcpy_wideport_out") {
      MemcpyWideport = true;
      propagatePort(WrapperPort::MEMORY_PORT);
      return true;
    }
  }

  if (n->getCalleeDecl()->hasAttr<OSSTaskDeclAttr>()) {
    CreatesTasks = true;
    propagatePort(WrapperPort::OUTPORT);
    return true;
  }

  auto *body = sym->getBody();
  if (!body) {
    return true;
  }

  FunctionCallTree newNode(sym, Current);
  FunctionCallTree *prev = Current;
  Current = &newNode;
  getDerived().TraverseStmt(body);
  Current = prev;
  return true;
}
