//===--- OmpSsKinds.h - OmpSs enums ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some OmpSs-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OMPSSKINDS_H
#define LLVM_CLANG_BASIC_OMPSSKINDS_H

#include "llvm/ADT/StringRef.h"

namespace clang {

/// OmpSs directives.
enum OmpSsDirectiveKind {
#define OMPSS_DIRECTIVE(Name) \
  OSSD_##Name,
#define OMPSS_DIRECTIVE_EXT(Name, Str) \
  OSSD_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSD_unknown
};

/// OmpSs clauses.
enum OmpSsClauseKind {
#define OMPSS_CLAUSE(Name, Class) \
  OSSC_##Name,
#define OMPSS_CLAUSE_ALIAS(Alias, Name) \
  OSSC_##Alias,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_unknown
};

/// OmpSs attributes for 'default' clause.
enum OmpSsDefaultClauseKind {
#define OMPSS_DEFAULT_KIND(Name) \
  OSSC_DEFAULT_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEFAULT_unknown
};

/// OmpSs attributes for 'depend' clause.
enum OmpSsDependClauseKind {
#define OMPSS_DEPEND_KIND(Name) \
  OSSC_DEPEND_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEPEND_unknown
};

OmpSsDirectiveKind getOmpSsDirectiveKind(llvm::StringRef Str);
const char *getOmpSsDirectiveName(OmpSsDirectiveKind Kind);

OmpSsClauseKind getOmpSsClauseKind(llvm::StringRef Str);
const char *getOmpSsClauseName(OmpSsClauseKind Kind);

unsigned getOmpSsSimpleClauseType(OmpSsClauseKind Kind, llvm::StringRef Str);
const char *getOmpSsSimpleClauseTypeName(OmpSsClauseKind Kind, unsigned Type);

bool isAllowedClauseForDirective(OmpSsDirectiveKind DKind,
                                 OmpSsClauseKind CKind);

/// Checks if the specified clause is one of private clauses like
/// 'private', 'firstprivate', 'reduction' etc..
/// \param Kind Clause kind.
/// \return true - the clause is a private clause, otherwise - false.
bool isOmpSsPrivate(OmpSsClauseKind Kind);

/// Checks if the specified directive kind is one of tasking directives:
/// task, task for,
/// taskloop or taskloop for.
bool isOmpSsTaskingDirective(OmpSsDirectiveKind Kind);

/// Checks if the specified directive is a directive with an associated
/// loop construct.
/// \param DKind Specified directive.
/// \return true - the directive is a loop-associated directive like 'oss taskloop'
/// or 'oss task for' directive, otherwise - false.
bool isOmpSsLoopDirective(OmpSsDirectiveKind DKind);

/// Checks if the specified directive is a directive with an associated
/// loop construct.
/// \param DKind Specified directive.
/// \return true - the directive is a loop-associated directive like 'oss taskloop'
/// or 'oss taskloop for' directive, otherwise - false.
bool isOmpSsTaskLoopDirective(OmpSsDirectiveKind DKind);

}

#endif
