//===--- OmpSsKinds.h - OmpSs enums ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Frontend/OmpSs/OSSConstants.h"

namespace clang {

/// OmpSs directives.
using OmpSsDirectiveKind = llvm::oss::Directive;

/// OmpSs clauses.
using OmpSsClauseKind = llvm::oss::Clause;

/// OmpSs attributes for 'depend' clause.
enum OmpSsDependClauseKind {
#define OMPSS_DEPEND_KIND(Name) \
  OSSC_DEPEND_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEPEND_unknown
};

/// OmpSs attributes for 'device' clause.
enum OmpSsDeviceClauseKind {
#define OMPSS_DEVICE_KIND(Name) \
  OSSC_DEVICE_##Name,
#include "clang/Basic/OmpSsKinds.def"
  OSSC_DEVICE_unknown
};

unsigned getOmpSsSimpleClauseType(OmpSsClauseKind Kind, llvm::StringRef Str);
const char *getOmpSsSimpleClauseTypeName(OmpSsClauseKind Kind, unsigned Type);

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
