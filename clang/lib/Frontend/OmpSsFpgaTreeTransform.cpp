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
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprOmpSs.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
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

  bool CreatesTasks;
  bool instrumented;

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

  QualType InstrPortType;
  IdentifierInfo *InstrPortIdentifier;
  ParmVarDecl *InstrPort;

  QualType McxxSetLockType;
  IdentifierInfo *McxxSetLockIdentifier;
  FunctionDecl *McxxSetLock;

  QualType McxxUnsetLockType;
  IdentifierInfo *McxxUnsetLockIdentifier;
  FunctionDecl *McxxUnsetLock;

  QualType McxxTaskwaitType;
  IdentifierInfo *McxxTaskwaitIdentifier;
  FunctionDecl *McxxTaskwait;

  QualType McxxTaskCreateType;
  IdentifierInfo *McxxTaskCreateIdentifier;
  FunctionDecl *McxxTaskCreate;

  QualType McxxInstrumentEventType;
  IdentifierInfo *McxxInstrumentEventIdentifier;
  FunctionDecl *McxxInstrumentEvent;

  bool needsDeps = false;

  ReplacementMap replMap;

  void addReplacementOpStmt(Stmt *OriginalPos, Stmt *Replacement) {
    replMap.addReplacementOpStmt(OriginalPos, Replacement);
  }

  QualType getTypeStr(StringRef str) {
    auto *decl = RecordDecl::Create(Ctx, TagTypeKind::TTK_Struct,
                                    Ctx.getTranslationUnitDecl(), {}, {},
                                    &IdentifierTable.get(str));
    return Ctx.getRecordType(decl);
  }

  DeclRefExpr *makeDeclRefExpr(const ValueDecl *Decl) const {
    assert(Decl && "Decl must not be null");
    return DeclRefExpr::Create(Ctx, NestedNameSpecifierLoc{}, SourceLocation{},
                               const_cast<ValueDecl *>(Decl), false,
                               SourceLocation{},
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
    return StringRef(mem, original.size());
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
        Ctx, Ctx.getTranslationUnitDecl(), {}, {},
        DeclarationName(FunctionIdentifier), FunctionType, nullptr, SC_None);

    return CallExpr::Create(Ctx, makeDeclRefExpr(FunctionDecl), parameters,
                            original->getType(), original->getValueKind(),
                            SourceLocation{}, original->getFPFeatures());
  }

  CallExpr *makeCallToFunc(FunctionDecl *FunctionDecl,
                           llvm::ArrayRef<Expr *> parameters) {
    llvm::SmallVector<QualType, 4> paramTypes(parameters.size());
    int i = 0;
    for (auto &&param : parameters) {
      paramTypes[i] = param->getType();
      ++i;
    }

    return CallExpr::Create(Ctx, makeDeclRefExpr(FunctionDecl), parameters,
                            FunctionDecl->getReturnType(), clang::VK_LValue,
                            SourceLocation{}, {});
  }

  MemberExpr *makeAccessExpr(Expr *Base, StringRef AccessMemberName) {
    auto *identifier = &IdentifierTable.get(AccessMemberName);

    auto declName = DeclarationName(identifier);
    FieldDecl *member =
        FieldDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {}, identifier,
                          Ctx.getIntPtrType(), nullptr, nullptr, true,
                          InClassInitStyle::ICIS_NoInit);
    return MemberExpr::Create(
        Ctx, Base, false, {}, {}, {}, member,
        DeclAccessPair::make(member, AccessSpecifier::AS_public),
        DeclarationNameInfo(declName, {}), nullptr, Ctx.getIntPtrType(),
        ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, NOUR_None);
  }

  DeclStmt *makeDeclStmt(VarDecl *declVar) {
    Decl *decl = static_cast<Decl *>(declVar);
    return new (Ctx) DeclStmt(DeclGroupRef::Create(Ctx, &decl, 1), {}, {});
  }

  VarDecl *makeVarDecl(QualType type, StringRef name) {
    auto *varDecl = VarDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {},
                                    &IdentifierTable.get(name), type, nullptr,
                                    StorageClass::SC_None);
    assert(
        varDecl
            ->getTranslationUnitDecl()); // Make sure we have a good declaration
                                         // context. Otherwise, things will
                                         // break in obscure ways.
    return varDecl;
  }

  Stmt *maybeMakeInstrumentedApiCall(Stmt *stmtToInstr,
                                     FPGAInstrumentationApiCalls call) {
    if (!instrumented) {
      return stmtToInstr;
    }
    llvm::SmallVector<Stmt *, 3> stmts;

    auto *instr = makeDeclRefExpr(InstrPort);
    stmts.push_back(makeCallToFunc(
        McxxInstrumentEvent,
        {makeIntegerLiteral(uint32_t(FPGAInstrumentationEvents::APICallBegin)),
         makeIntegerLiteral(uint64_t(call)), instr}));

    stmts.push_back(stmtToInstr);

    stmts.push_back(makeCallToFunc(
        McxxInstrumentEvent,
        {makeIntegerLiteral(uint32_t(FPGAInstrumentationEvents::APICallEnd)),
         makeIntegerLiteral(uint64_t(call)), instr}));

    auto *stmt = CompoundStmt::Create(Ctx, stmts, FPOptionsOverride(),
                                      SourceLocation{}, SourceLocation{});

    return stmt;
  }

  std::string typeToString(QualType type) {
    std::string outType;
    llvm::raw_string_ostream out(outType);
    type.print(out, PrintPol);
    return outType;
  }

public:
  OmpSsFpgaTreeTransformVisitor(ASTContext &Ctx,
                                clang::IdentifierTable &IdentifierTable,
                                ::WrapperPortMap &WrapperPortMap,
                                uint64_t FpgaPortWidth, bool CreatesTasks,
                                bool instrumented)
      : Inherited(), Ctx(Ctx), IdentifierTable(IdentifierTable),
        WrapperPortMap(WrapperPortMap), PrintPol(Ctx.getLangOpts()),
        CreatesTasks(CreatesTasks), instrumented(instrumented) {
    OmpIfRankType = Ctx.UnsignedCharTy;
    OmpIfRankIdentifier = &IdentifierTable.get("__ompif_rank");

    OmpIfSizeType = Ctx.UnsignedCharTy;
    OmpIfSizeIdentifier = &IdentifierTable.get("__ompif_size");

    SpawnInPortType =
        Ctx.getLValueReferenceType(getTypeStr("hls::stream<ap_uint<8> >"));
    SpawnInPortIdentifier = &IdentifierTable.get("mcxx_spawnInPort");

    InPortType =
        Ctx.getLValueReferenceType(getTypeStr("hls::stream<ap_uint<64> >"));
    InPortIdentifier = &IdentifierTable.get("mcxx_inPort");

    OutPortType =
        Ctx.getLValueReferenceType(getTypeStr("hls::stream<mcxx_outaxis>"));
    OutPortIdentifier = &IdentifierTable.get("mcxx_outPort");

    MemPortType = Ctx.getPointerType(getTypeStr(
        AllocatedStringRef("ap_uint<" + std::to_string(FpgaPortWidth) + ">")));
    MemPortIdentifier = &IdentifierTable.get("mcxx_memport");

    InstrPortType = Ctx.getLValueReferenceType(
        getTypeStr("hls::stream<__mcxx_instrData_t>"));
    InstrPortIdentifier = &IdentifierTable.get("mcxx_instr");

    McxxSetLockType = Ctx.getFunctionType(Ctx.VoidTy, {InPortType, OutPortType},
                                          FunctionProtoType::ExtProtoInfo());
    McxxSetLockIdentifier = &IdentifierTable.get("mcxx_set_lock");
    McxxSetLock =
        FunctionDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {},
                             DeclarationName(McxxSetLockIdentifier),
                             McxxSetLockType, nullptr, SC_None);

    McxxUnsetLockType = Ctx.getFunctionType(Ctx.VoidTy, {OutPortType},
                                            FunctionProtoType::ExtProtoInfo());
    McxxUnsetLockIdentifier = &IdentifierTable.get("mcxx_unset_lock");
    McxxUnsetLock =
        FunctionDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {},
                             DeclarationName(McxxUnsetLockIdentifier),
                             McxxUnsetLockType, nullptr, SC_None);

    McxxTaskwaitType =
        Ctx.getFunctionType(Ctx.VoidTy, {SpawnInPortType, OutPortType},
                            FunctionProtoType::ExtProtoInfo());
    McxxTaskwaitIdentifier = &IdentifierTable.get("mcxx_taskwait");
    McxxTaskwait =
        FunctionDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {},
                             DeclarationName(McxxTaskwaitIdentifier),
                             McxxTaskwaitType, nullptr, SC_None);

    McxxTaskCreateType = Ctx.getFunctionType(
        Ctx.VoidTy,
        {Ctx.UnsignedLongLongTy, Ctx.UnsignedLongLongTy, Ctx.UnsignedLongLongTy,
         Ctx.VoidPtrTy, Ctx.UnsignedLongLongTy, Ctx.VoidPtrTy,
         Ctx.UnsignedLongLongTy, Ctx.VoidPtrTy, OutPortType},
        {});
    McxxTaskCreateIdentifier = &IdentifierTable.get("mcxx_task_create");
    McxxTaskCreate =
        FunctionDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {},
                             DeclarationName(McxxTaskCreateIdentifier),
                             McxxTaskCreateType, nullptr, SC_None);

    McxxInstrumentEventType = Ctx.getFunctionType(
        Ctx.VoidTy, {Ctx.UnsignedCharTy, Ctx.UnsignedLongLongTy, InstrPortType},
        FunctionProtoType::ExtProtoInfo());
    McxxInstrumentEventIdentifier =
        &IdentifierTable.get("mcxx_instrument_event");
    McxxInstrumentEvent =
        FunctionDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), {}, {},
                             DeclarationName(McxxInstrumentEventIdentifier),
                             McxxInstrumentEventType, nullptr, SC_None);
  }

  ReplacementMap &&takeReplacementMap() { return std::move(replMap); }

  bool getNeedsDeps() { return needsDeps; }

  bool VisitVarDecl(VarDecl *decl) {
    QualType type = decl->getType();
    if (CreatesTasks && type->isPointerType()) {
      type = type->getPointeeType().IgnoreParens();
      type = getTypeStr(
          AllocatedStringRef("__mcxx_ptr_t<" + typeToString(type) + " >"));
      decl->setType(type);
      decl->setTypeSourceInfo(nullptr);
    }
    return true;
  }

  llvm::SmallVector<Stmt *, 1> copyParamTaskCall(
      llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo> &copies,
      VarDecl *copiesDecl, ParmVarDecl *param, int &copId, int paramId,
      Expr *accessedMember, QualType typeMember) {

    llvm::SmallVector<Stmt *, 1> stmts;

    auto copIt = copies.find(param);
    if (copIt != copies.end()) {
      auto copy = *copIt;
      Expr *copyIdExpr = makeIntegerLiteral(copId);

      auto *copiesSubscript = new (Ctx) ArraySubscriptExpr(
          makeDeclRefExpr(copiesDecl), copyIdExpr,
          QualType(copiesDecl->getType()->getPointeeOrArrayElementType(), 0),
          ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {});

      stmts.push_back(BinaryOperator::Create(
          Ctx, makeAccessExpr(copiesSubscript, "copy_address"), accessedMember,
          BinaryOperator::Opcode::BO_Assign, typeMember,
          ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}));

      stmts.push_back(BinaryOperator::Create(
          Ctx, makeAccessExpr(copiesSubscript, "arg_idx"),
          makeIntegerLiteral(paramId), BinaryOperator::Opcode::BO_Assign,
          typeMember, ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {},
          {}));

      stmts.push_back(BinaryOperator::Create(
          Ctx, makeAccessExpr(copiesSubscript, "flags"),
          makeIntegerLiteral(unsigned(copy.second.dir)),
          BinaryOperator::Opcode::BO_Assign, typeMember,
          ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}));

      auto *sizeArray = [&] {
        if (copy.second.FixedArrayRef) {
          return makeIntegerLiteral(ComputeArrayRefSize(
              Ctx, copy.second.FixedArrayRef,
              Ctx.getTypeSize(GetElementTypePointerTo(copy.first->getType())) /
                  Ctx.getCharWidth()));
        }
        // UnaryExprOrTypeTraitExpr e(UnaryExprOrTypeTrait::UETT_SizeOf, );
        return makeIntegerLiteral(
            Ctx.getTypeSize(GetElementTypePointerTo(copy.first->getType())) /
            Ctx.getCharWidth());
      }();

      stmts.push_back(BinaryOperator::Create(
          Ctx, makeAccessExpr(copiesSubscript, "size"), sizeArray,
          BinaryOperator::Opcode::BO_Assign, typeMember,
          ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}));

      ++copId;
    }
    return stmts;
  }

  bool TaskCall(CallExpr *callExpr, OSSTaskDeclAttr *attr) {
    auto dependencyMap = computeDependencyMap(attr, true);

    llvm::SmallVector<Stmt *, 1> stmts;

    VarDecl *argsDecl = nullptr;
    {
      auto typeArgs = Ctx.getConstantArrayType(
          Ctx.UnsignedLongLongTy, llvm::APInt(64, callExpr->getNumArgs()),
          makeIntegerLiteral(callExpr->getNumArgs()),
          ArrayType::ArraySizeModifier::Normal, 0);
      argsDecl = makeVarDecl(typeArgs, "__mcxx_args");
      stmts.push_back(makeDeclStmt(argsDecl));
    }

    llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo> copies;
    if (attr->getDevice() == OSSTaskDeclAttr::Fpga)
      copies =
          ComputeLocalmems(dyn_cast<FunctionDecl>(callExpr->getCalleeDecl()));
    else {
      for (auto &&[param, dependency] : dependencyMap) {
        if (auto *arrExpr = dyn_cast<OSSArrayShapingExpr>(dependency.first)) {
          copies.insert(std::pair<const ParmVarDecl *, LocalmemInfo>(
              param, LocalmemInfo{-1, arrExpr, dependency.second}));
        } else {
          copies.insert(std::pair<const ParmVarDecl *, LocalmemInfo>(
              param, LocalmemInfo{-1, nullptr, dependency.second}));
        }
      }
    }

    VarDecl *depsDecl = nullptr;
    if (!dependencyMap.empty()) {
      auto typeArgs = Ctx.getConstantArrayType(
          Ctx.UnsignedLongLongTy, llvm::APInt(64, dependencyMap.size()),
          makeIntegerLiteral(dependencyMap.size()),
          ArrayType::ArraySizeModifier::Normal, 0);

      depsDecl = makeVarDecl(typeArgs, "__mcxx_deps");

      stmts.push_back(makeDeclStmt(depsDecl));
    }

    VarDecl *copiesDecl = nullptr;
    if (!copies.empty()) {
      auto typeArgs = Ctx.getConstantArrayType(
          getTypeStr("__fpga_copyinfo_t"), llvm::APInt(64, copies.size()),
          makeIntegerLiteral(copies.size()),
          ArrayType::ArraySizeModifier::Normal, 0);

      copiesDecl = makeVarDecl(typeArgs, "__mcxx_copies");
      stmts.push_back(makeDeclStmt(copiesDecl));
    }

    int paramId = 0;
    int copId = 0;
    int depId = 0;
    for (auto [argIt, paramIt, argEnd, paramEnd] =
             std::tuple{callExpr->arg_begin(),
                        dyn_cast<FunctionDecl>(callExpr->getCalleeDecl())
                            ->param_begin(),
                        callExpr->arg_end(),
                        dyn_cast<FunctionDecl>(callExpr->getCalleeDecl())
                            ->param_end()};
         argIt != argEnd && paramIt != paramEnd; ++argIt, ++paramIt) {
      Expr *arg = *argIt;
      ParmVarDecl *param = *paramIt;

      QualType paramType = param->getType();

      VarDecl *classInstance;
      Expr *accessedMember;
      if (paramType->isPointerType()) {
        QualType ptrType = getTypeStr(AllocatedStringRef(
            "__mcxx_ptr_t<" + typeToString(paramType->getPointeeType()) +
            " >"));

        classInstance =
            makeVarDecl(ptrType, AllocatedStringRef("__mcxx_arg_" +
                                                    std::to_string(paramId)));
        stmts.push_back(makeDeclStmt(classInstance));

        stmts.push_back(BinaryOperator::Create(
            Ctx, makeDeclRefExpr(classInstance), arg,
            BinaryOperator::Opcode::BO_Assign, ptrType,
            ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}));

        accessedMember = makeAccessExpr(makeDeclRefExpr(classInstance), "val");
      } else {
        paramType.removeLocalCVRQualifiers(Qualifiers::CVRMask);
        QualType castUnionType = getTypeStr(AllocatedStringRef(
            "__mcxx_cast<" + typeToString(paramType) + " >"));

        classInstance = makeVarDecl(
            castUnionType,
            AllocatedStringRef("cast_param_" + std::to_string(paramId)));
        stmts.push_back(makeDeclStmt(classInstance));

        stmts.push_back(BinaryOperator::Create(
            Ctx, makeAccessExpr(makeDeclRefExpr(classInstance), "typed"), arg,
            BinaryOperator::Opcode::BO_Assign, castUnionType,
            ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}));

        accessedMember = makeAccessExpr(makeDeclRefExpr(classInstance), "raw");
      }

      auto typeMember =
          QualType(argsDecl->getType()->getPointeeOrArrayElementType(), 0);
      stmts.push_back(BinaryOperator::Create(
          Ctx,
          new (Ctx) ArraySubscriptExpr(makeDeclRefExpr(argsDecl),
                                       makeIntegerLiteral(paramId), typeMember,
                                       ExprValueKind::VK_LValue,
                                       ExprObjectKind::OK_Ordinary, {}),
          accessedMember, BinaryOperator::Opcode::BO_Assign, typeMember,
          ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}));

      stmts.append(copyParamTaskCall(copies, copiesDecl, param, copId, paramId,
                                     accessedMember, typeMember));

      if (auto dependIt = dependencyMap.find(param);
          dependIt != dependencyMap.end()) {
        needsDeps = true;

        auto *flagExpression = BinaryOperator::Create(
            Ctx, makeIntegerLiteral(uint64_t(dependIt->second.second)),
            makeIntegerLiteral(58ULL), BO_Shl, Ctx.UnsignedIntTy,
            ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {});

        auto paramType = param->getType();
        auto dataType = paramType;
        if (dataType->isPointerType()) {
          dataType = dataType->getPointeeType();
        }
        auto type = getTypeStr(AllocatedStringRef(
            "__mcxx_ptr_t<" + typeToString(dataType) + " >"));

        auto *ptrVar = makeVarDecl(
            type, AllocatedStringRef("__mcxx_dep_" + std::to_string(depId)));

        stmts.push_back(makeDeclStmt(ptrVar));

        /* Figure out what this was doing

        stmts.append(BinaryOperator::Create(
            Ctx, makeDeclRefExpr(ptrVar),
            BinaryOperator::Create(
                Ctx, Expr * lhs, Expr * rhs, BinaryOperatorKind::BO_Add, type,
                ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}, {}),
            BinaryOperatorKind::BO_Assign, type, ExprValueKind::VK_LValue,
            ExprObjectKind::OK_Ordinary, {}, {}));

        spawn_nodes.append(
            Nodecl::ExpressionStatement::make(Nodecl::Assignment::make(
                mcxx_ptr_var.make_nodecl(),
                Nodecl::Add::make(
                    base_address,
                    Nodecl::Div::make(
                        dep.ref.get_offsetof_dependence(),
                        const_value_to_nodecl(
                            const_value_get_unsigned_int(data_type.get_size())),
                        Type::get_unsigned_long_long_int_type()),
                    Type::get_unsigned_long_long_int_type()),
                mcxx_ptr_var.get_type())));*/

        // This is a stub until I figure out what the code above was doing.
        stmts.push_back(BinaryOperator::Create(
            Ctx, makeDeclRefExpr(ptrVar), arg, BinaryOperatorKind::BO_Assign,
            type, ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {},
            {}));
        stmts.push_back(BinaryOperator::Create(
            Ctx,
            new (Ctx) ArraySubscriptExpr(
                makeDeclRefExpr(depsDecl), makeIntegerLiteral(depId),
                QualType(depsDecl->getType()->getPointeeOrArrayElementType(),
                         0),
                ExprValueKind::VK_LValue, ExprObjectKind::OK_Ordinary, {}),
            BinaryOperator::Create(
                Ctx, flagExpression,
                makeAccessExpr(makeDeclRefExpr(ptrVar), "val"),
                BinaryOperatorKind::BO_Or, type, ExprValueKind::VK_LValue,
                ExprObjectKind::OK_Ordinary, {}, {}),
            BinaryOperatorKind::BO_Assign, type, ExprValueKind::VK_LValue,
            ExprObjectKind::OK_Ordinary, {}, {}));
        ++depId;
      }
      ++paramId;
    }

    llvm::SmallVector<Expr *, 10> arguments;

    arguments.push_back(makeIntegerLiteral(
        GenOnto(Ctx, dyn_cast<FunctionDecl>(callExpr->getCalleeDecl()))));
    if (auto *affinity = attr->getAffinity()) {
      arguments.push_back(affinity);
    } else {
      arguments.push_back(makeIntegerLiteral(0xFF));
    }

    arguments.push_back(makeIntegerLiteral(paramId));
    arguments.push_back(makeDeclRefExpr(argsDecl));

    arguments.push_back(makeIntegerLiteral(depId));
    if (depId > 0) {
      arguments.push_back(makeDeclRefExpr(depsDecl));
    } else {
      arguments.push_back(makeIntegerLiteral(0));
    }

    arguments.push_back(makeIntegerLiteral(copId));
    if (copId > 0) {
      arguments.push_back(makeDeclRefExpr(copiesDecl));
    } else {
      arguments.push_back(makeIntegerLiteral(0));
    }

    arguments.push_back(makeDeclRefExpr(OutPort));

    stmts.push_back(
        maybeMakeInstrumentedApiCall(makeCallToFunc(McxxTaskCreate, arguments),
                                     FPGAInstrumentationApiCalls::CreateTask));

    addReplacementOpStmt(callExpr,
                         CompoundStmt::Create(Ctx, stmts, {}, {}, {}));

    return true;
  }

  bool VisitCallExpr(CallExpr *callExpr) {
    if (auto *attr = callExpr->getCalleeDecl()->getAttr<OSSTaskDeclAttr>()) {
      return TaskCall(callExpr, attr);
    }

    const StringRef funcName =
        dyn_cast<FunctionDecl>(callExpr->getCalleeDecl())->getName();

    if (funcName == "OMPIF_Comm_rank") {
      auto *operation = BinaryOperator::Create(
          Ctx, makeDeclRefExpr(OmpIfRank),
          UnaryOperator::Create(
              Ctx, makeIntegerLiteral(1), UnaryOperatorKind::UO_Minus,
              Ctx.IntTy, ExprValueKind::VK_PRValue, ExprObjectKind::OK_Ordinary,
              SourceLocation{}, false, {}),
          BinaryOperatorKind::BO_Add, Ctx.IntTy, callExpr->getValueKind(),
          callExpr->getObjectKind(), SourceLocation{}, {});
      auto *paren =
          new (Ctx) ParenExpr(SourceLocation{}, SourceLocation{}, operation);
      addReplacementOpStmt(callExpr, paren);
      return true;
    }
    if (funcName == "OMPIF_Comm_size") {
      addReplacementOpStmt(callExpr, makeDeclRefExpr(OmpIfRank));
      return true;
    }
    if (funcName == "OMPIF_Send" || funcName == "OMPIF_Recv") {
      llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
      arguments.reserve(callExpr->getNumArgs() + 3);
      arguments.push_back(makeIntegerLiteral(0));
      arguments.push_back(makeIntegerLiteral(0));
      arguments.push_back(makeDeclRefExpr(OutPort));

      addReplacementOpStmt(callExpr,
                           makeCallToWithDifferentParams(callExpr, arguments));
      return true;
    }
    if (funcName == "OMPIF_Allgather") {
      llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
      arguments.reserve(callExpr->getNumArgs() + 3);
      arguments.push_back(makeDeclRefExpr(OmpIfRank));
      arguments.push_back(makeDeclRefExpr(SpawnInPort));
      arguments.push_back(makeDeclRefExpr(OutPort));

      addReplacementOpStmt(callExpr,
                           makeCallToWithDifferentParams(callExpr, arguments));
      return true;
    }
    if (funcName == "nanos6_fpga_memcpy_wideport_in" ||
        funcName == "nanos6_fpga_memcpy_wideport_out") {
      llvm::SmallVector<Expr *, 4> arguments(callExpr->arguments());
      arguments.push_back(makeDeclRefExpr(MemPort));
      addReplacementOpStmt(callExpr,
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
    if (instrumented)
      arguments.push_back(makeDeclRefExpr(InstrPort));
    addReplacementOpStmt(callExpr,
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

    auto &wrapperMap = WrapperPortMap[funcDecl->getCanonicalDecl()];

    OmpIfRank = (wrapperMap[(int)WrapperPort::OMPIF_RANK])
                    ? addNewParamInfo(OmpIfRankIdentifier, OmpIfRankType)
                    : nullptr;
    OmpIfSize = (wrapperMap[(int)WrapperPort::OMPIF_SIZE])
                    ? addNewParamInfo(OmpIfSizeIdentifier, OmpIfSizeType)
                    : nullptr;
    SpawnInPort = (wrapperMap[(int)WrapperPort::SPAWN_INPORT])
                      ? addNewParamInfo(SpawnInPortIdentifier, SpawnInPortType)
                      : nullptr;
    InPort = (wrapperMap[(int)WrapperPort::INPORT])
                 ? addNewParamInfo(InPortIdentifier, InPortType)
                 : nullptr;
    OutPort = (wrapperMap[(int)WrapperPort::OUTPORT])
                  ? addNewParamInfo(OutPortIdentifier, OutPortType)
                  : nullptr;
    MemPort = (wrapperMap[(int)WrapperPort::MEMORY_PORT])
                  ? addNewParamInfo(MemPortIdentifier, MemPortType)
                  : nullptr;

    InstrPort = (instrumented)
                    ? addNewParamInfo(InstrPortIdentifier, InstrPortType)
                    : nullptr;

    const auto *origType = funcDecl->getType()->getAs<FunctionType>();
    llvm::SmallVector<QualType, 4> typesParam;
    typesParam.reserve(NewParamInfo.size());
    for (auto *param : NewParamInfo) {
      typesParam.push_back(param->getType());
    }

    if (origType->isFunctionProtoType()) {
      auto *funcType = origType->getAs<FunctionProtoType>();
      funcDecl->setType(Ctx.getFunctionType(
          funcType->getReturnType(), typesParam, funcType->getExtProtoInfo()));
      funcDecl->resetParams();
      funcDecl->setParams(NewParamInfo);
    } else if (origType->isFunctionNoProtoType()) {
      auto *funcType = origType->getAs<FunctionNoProtoType>();
      funcDecl->setType(
          Ctx.getFunctionType(funcType->getReturnType(), typesParam, {}));
    } else {
      llvm_unreachable("Other types of function are not implemented (if any?)");
    }
    funcDecl->resetParams();
    funcDecl->setParams(NewParamInfo);
    funcDecl->setHasWrittenPrototype();

    return true;
  }

  bool VisitOSSCriticalDirective(OSSCriticalDirective *n) {
    llvm::SmallVector<Stmt *, 4> stmts;

    DeclRefExpr *in = makeDeclRefExpr(InPort), *out = makeDeclRefExpr(OutPort);
    stmts.push_back(
        maybeMakeInstrumentedApiCall(makeCallToFunc(McxxSetLock, {in, out}),
                                     FPGAInstrumentationApiCalls::SetLock));

    if (n->hasAssociatedStmt())
      stmts.push_back(n->getAssociatedStmt());

    stmts.push_back(
        maybeMakeInstrumentedApiCall(makeCallToFunc(McxxUnsetLock, {out}),
                                     FPGAInstrumentationApiCalls::UnsetLock));

    auto *stmt = CompoundStmt::Create(Ctx, stmts, FPOptionsOverride(),
                                      SourceLocation{}, SourceLocation{});
    addReplacementOpStmt(n, stmt);
    return true;
  }

  bool VisitOSSTaskwaitDirective(OSSTaskwaitDirective *n) {
    DeclRefExpr *SapawnIn = makeDeclRefExpr(SpawnInPort),
                *out = makeDeclRefExpr(OutPort);

    addReplacementOpStmt(n, maybeMakeInstrumentedApiCall(
                                makeCallToFunc(McxxTaskwait, {SapawnIn, out}),
                                FPGAInstrumentationApiCalls::WaitTasks));
    return true;
  }
};

unsigned int MercuriumHashStr(const char *str) {

  const int MULTIPLIER = 33;
  unsigned int h;
  unsigned const char *p;

  h = 0;
  for (p = (unsigned const char *)str; *p != '\0'; p++)
    h = MULTIPLIER * h + *p;

  h += (h >> 5);
  return h; // or, h % ARRAY_SIZE;
}

} // namespace
namespace clang {
std::pair<bool, ReplacementMap>
OmpssFpgaTreeTransform(clang::ASTContext &Ctx,
                       clang::IdentifierTable &identifierTable,
                       WrapperPortMap &WrapperPortMap, uint64_t FpgaPortWidth,
                       bool CreatesTasks, bool instrumented) {
  OmpSsFpgaTreeTransformVisitor t(Ctx, identifierTable, WrapperPortMap,
                                  FpgaPortWidth, CreatesTasks, instrumented);
  t.TraverseAST(Ctx);
  return {t.getNeedsDeps(), t.takeReplacementMap()};
}

ParamDependencyMap computeDependencyMap(OSSTaskDeclAttr *taskAttr,
                                        bool includeNonArrays) {
  ParamDependencyMap currentAssignationsOfArrays;

  auto GetTheArgument = [&](const Expr *OSSExpr) -> const ParmVarDecl * {
    if (auto *arrShapingExpr = dyn_cast<OSSArrayShapingExpr>(OSSExpr)) {
      auto *arrExprBase = dyn_cast<DeclRefExpr>(
          arrShapingExpr->getBase()->IgnoreParenImpCasts());
      assert(arrExprBase);
      if (!arrExprBase)
        return nullptr;
      auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
      assert(decl);
      return decl;
    }
    if (auto *arrSectionExpr = dyn_cast<OSSArraySectionExpr>(OSSExpr);
        includeNonArrays && arrSectionExpr) {
      auto *arrExprBase = dyn_cast<DeclRefExpr>(
          arrSectionExpr->getBase()->IgnoreParenImpCasts());
      assert(arrExprBase);
      if (!arrExprBase)
        return nullptr;
      auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
      assert(decl);
      return decl;
    }
    if (auto *MultiDepExpr = dyn_cast<OSSMultiDepExpr>(OSSExpr);
        includeNonArrays && MultiDepExpr) {
      auto *arrExprBase = dyn_cast<DeclRefExpr>(
          MultiDepExpr->getDepExpr()->IgnoreParenImpCasts());
      if (!arrExprBase)
        return nullptr;
      auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
      return decl;
    }
    if (auto *exprDecl = dyn_cast<DeclRefExpr>(OSSExpr->IgnoreParenImpCasts());
        includeNonArrays && exprDecl) {
      return dyn_cast<ParmVarDecl>(exprDecl->getDecl());
    }
    return nullptr;
  };

  auto EmitDepListIterDecls = [&](auto &&DepExprsIter, LocalmemInfo::Dir dir) {
    for (const Expr *DepExpr : DepExprsIter) {
      auto *decl = GetTheArgument(DepExpr);
      if (!decl)
        continue;
      auto res = currentAssignationsOfArrays.find(decl);
      if (res != currentAssignationsOfArrays.end()) {
        res->second.second = LocalmemInfo::Dir(res->second.second | dir);
      } else {
        currentAssignationsOfArrays.insert({decl, {DepExpr, dir}});
      };
    }
  };
  EmitDepListIterDecls(taskAttr->ins(), LocalmemInfo::IN);
  EmitDepListIterDecls(taskAttr->outs(), LocalmemInfo::OUT);
  EmitDepListIterDecls(taskAttr->inouts(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->concurrents(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->weakIns(), LocalmemInfo::IN);
  EmitDepListIterDecls(taskAttr->weakOuts(), LocalmemInfo::OUT);
  EmitDepListIterDecls(taskAttr->weakInouts(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->weakConcurrents(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->depIns(), LocalmemInfo::IN);
  EmitDepListIterDecls(taskAttr->depOuts(), LocalmemInfo::OUT);
  EmitDepListIterDecls(taskAttr->depInouts(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->depConcurrents(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->depWeakIns(), LocalmemInfo::IN);
  EmitDepListIterDecls(taskAttr->depWeakOuts(), LocalmemInfo::OUT);
  EmitDepListIterDecls(taskAttr->depWeakInouts(), LocalmemInfo::INOUT);
  EmitDepListIterDecls(taskAttr->depWeakConcurrents(), LocalmemInfo::INOUT);
  return currentAssignationsOfArrays;
}

llvm::SmallDenseMap<const clang::ParmVarDecl *, LocalmemInfo>
ComputeLocalmems(FunctionDecl *FD) {
  auto *taskAttr = FD->getAttr<OSSTaskDeclAttr>();
  // First, compute the direction tags of the parameters. Do note that not
  // all parameters are guaranteed to be present
  ParamDependencyMap currentAssignationsOfArrays =
      computeDependencyMap(taskAttr);

  // Then compute the list of localmem parameters
  llvm::SmallDenseSet<const ParmVarDecl *> parametersToLocalmem;

  // If copy_deps
  if (taskAttr->getCopyDeps()) {
    for (auto *param : FD->parameters()) {
      if (currentAssignationsOfArrays.find(param) !=
          currentAssignationsOfArrays.end()) {
        parametersToLocalmem.insert(param);
      }
    }
  }
  // If we have an explicit list of localmem (copy_in, copy_out, copy_inout),
  // use that
  auto explicitCopy = [&](auto &&list, LocalmemInfo::Dir dir) {
    for (auto *localmem : list) {
      auto *arrShapingExpr = dyn_cast<OSSArrayShapingExpr>(localmem);
      if (!arrShapingExpr) {
        llvm_unreachable("We have checked this in a sema pass");
        continue;
      }

      auto *arrExprBase = dyn_cast<DeclRefExpr>(
          arrShapingExpr->getBase()->IgnoreParenImpCasts());
      assert(arrExprBase);
      auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
      assert(decl);
      parametersToLocalmem.insert(decl);
      auto &def = currentAssignationsOfArrays[decl];
      def.first = arrShapingExpr;
      def.second = LocalmemInfo::Dir(def.second | dir);
    }
  };
  explicitCopy(taskAttr->copyIn(), LocalmemInfo::IN);
  explicitCopy(taskAttr->copyOut(), LocalmemInfo::OUT);
  explicitCopy(taskAttr->copyInOut(), LocalmemInfo::INOUT);

  // Compute the localmem list
  llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo> localmemList;
  for (auto *param : parametersToLocalmem) {
    auto data = currentAssignationsOfArrays.find(param);
    localmemList.insert(
        {param,
         LocalmemInfo{-1, dyn_cast<OSSArrayShapingExpr>(data->second.first),
                      data->second.second}});
  }
  return localmemList;
}

QualType DerefOnceTypePointerTo(QualType type) {
  if (type->isPointerType()) {
    return type->getPointeeType().IgnoreParens();
  }
  if (type->isArrayType()) {
    return type->getAsArrayTypeUnsafe()->getElementType().IgnoreParens();
  }
  return type;
}

QualType GetElementTypePointerTo(QualType type) {
  QualType pointsTo = type;

  for (auto isPointer = pointsTo->isPointerType(),
            isArray = pointsTo->isArrayType();
       isPointer || isArray; isPointer = pointsTo->isPointerType(),
            isArray = pointsTo->isArrayType()) {
    if (isPointer) {
      pointsTo = pointsTo->getPointeeType().IgnoreParens();
    } else if (isArray) {
      pointsTo =
          pointsTo->getAsArrayTypeUnsafe()->getElementType().IgnoreParens();
    }
  }

  return pointsTo;
}

QualType LocalmemArrayType(ASTContext &Ctx,
                           const OSSArrayShapingExpr *arrayType) {
  auto paramType = arrayType->getBase()->getType();
  for (size_t i = 0; i < arrayType->getShapes().size(); ++i)
    paramType = DerefOnceTypePointerTo(paramType);
  paramType.removeLocalCVRQualifiers(Qualifiers::CVRMask);
  for (auto *shape : arrayType->getShapes()) {
    auto computedSize = shape->getIntegerConstantExpr(Ctx);
    if (!computedSize) {
      llvm_unreachable(
          "We have already checked that the shape expressions evaluate to "
          "positive integers, we should be able to use them here safely");
    }
    paramType = Ctx.getConstantArrayType(paramType, *computedSize, shape,
                                         ArrayType::ArraySizeModifier{}, 0);
  }
  return paramType;
}

uint64_t ComputeArrayRefSize(ASTContext &Ctx,
                             const OSSArrayShapingExpr *arrayType,
                             uint64_t baseType) {
  uint64_t totalSize = baseType;
  auto paramType = LocalmemArrayType(Ctx, arrayType);
  while (paramType->isArrayType()) {
    if (!paramType->isConstantArrayType()) {
      llvm_unreachable("We have already checked that the parameter type is a "
                       "constant array");
    }
    auto *arrType = dyn_cast<ConstantArrayType>(paramType);
    paramType = DerefOnceTypePointerTo(paramType);
    totalSize *= arrType->getSize().getZExtValue();
  }

  return totalSize;
}

uint64_t GenOnto(ASTContext &Ctx, FunctionDecl *FD) {
  auto *taskAttr = FD->getAttr<OSSTaskDeclAttr>();
  auto *ontoExpr = taskAttr->getOnto();
  // Check onto information
  if (ontoExpr) {
    auto ontoRes = ontoExpr->getIntegerConstantExpr(Ctx);
    if (ontoRes && *ontoRes >= 0) {
      uint64_t onto = ontoRes->getZExtValue();
      // Check that arch bits are set
      if ((onto & 0x300000000) != 0 && onto <= 0x3FFFFFFFF) {
        return onto;
      }
    }
  }
  // Not using the line number to allow future modifications of source code
  // without afecting the accelerator hash
  std::string typeStr;
  llvm::raw_string_ostream typeStream(typeStr);
  std::unique_ptr<MangleContext> MC;
  MC.reset(Ctx.createMangleContext());
  if (MC->shouldMangleDeclName(FD)) {
    MC->mangleName(GlobalDecl(FD), typeStream);
  } else {
    typeStr = FD->getNameAsString();
  }
  unsigned long long int type = MercuriumHashStr(typeStr.c_str()) &
                                0xFFFFFFFF; //< Ensure that it its upto 32b
  if (taskAttr->getDevice() == OSSTaskDeclAttr::Fpga) {
    // FPGA flag
    type |= 0x100000000;
  } else {
    // SMP flag
    type |= 0x200000000;
  }
  return type;
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
