//===--- OmpSsKinds.cpp - Token Kinds Support ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the OmpSs enum and support functions.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang;

OmpSsDirectiveKind clang::getOmpSsDirectiveKind(StringRef Str) {
  return llvm::StringSwitch<OmpSsDirectiveKind>(Str)
#define OMPSS_DIRECTIVE(Name) .Case(#Name, OSSD_##Name)
#define OMPSS_DIRECTIVE_EXT(Name, Str) .Case(Str, OSSD_##Name)
#include "clang/Basic/OmpSsKinds.def"
      .Default(OSSD_unknown);
}

const char *clang::getOmpSsDirectiveName(OmpSsDirectiveKind Kind) {
  assert(Kind <= OSSD_unknown);
  switch (Kind) {
  case OSSD_unknown:
    return "unknown";
#define OMPSS_DIRECTIVE(Name)                                                 \
  case OSSD_##Name:                                                            \
    return #Name;
#define OMPSS_DIRECTIVE_EXT(Name, Str)                                        \
  case OSSD_##Name:                                                            \
    return Str;
#include "clang/Basic/OmpSsKinds.def"
    break;
  }
  llvm_unreachable("Invalid OmpSs directive kind");
}

OmpSsClauseKind clang::getOmpSsClauseKind(StringRef Str) {
  return llvm::StringSwitch<OmpSsClauseKind>(Str)
#define OMPSS_CLAUSE(Name, Class) .Case(#Name, OSSC_##Name)
#define OMPSS_CLAUSE_ALIAS(Alias, Name) .Case(#Alias, OSSC_##Alias)
#include "clang/Basic/OmpSsKinds.def"
      .Default(OSSC_unknown);
}

const char *clang::getOmpSsClauseName(OmpSsClauseKind Kind) {
  assert(Kind <= OSSC_unknown);
  switch (Kind) {
  case OSSC_unknown:
    return "unknown";
#define OMPSS_CLAUSE(Name, Class)                                             \
  case OSSC_##Name:                                                            \
    return #Name;
#define OMPSS_CLAUSE_ALIAS(Alias, Name)                                       \
  case OSSC_##Alias:                                                           \
    return #Alias;
#include "clang/Basic/OmpSsKinds.def"
  }
  llvm_unreachable("Invalid OmpSs clause kind");
}

unsigned clang::getOmpSsSimpleClauseType(OmpSsClauseKind Kind,
                                         StringRef Str) {
  switch (Kind) {
  case OSSC_default:
    return llvm::StringSwitch<OmpSsDefaultClauseKind>(Str)
#define OMPSS_DEFAULT_KIND(Name) .Case(#Name, OSSC_DEFAULT_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEFAULT_unknown);
  case OSSC_depend:
    return llvm::StringSwitch<OmpSsDependClauseKind>(Str)
#define OMPSS_DEPEND_KIND(Name) .Case(#Name, OSSC_DEPEND_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEPEND_unknown);
  case OSSC_unknown:
  case OSSC_if:
  case OSSC_final:
  case OSSC_cost:
  case OSSC_priority:
  case OSSC_label:
  case OSSC_wait:
  case OSSC_onready:
  case OSSC_private:
  case OSSC_firstprivate:
  case OSSC_shared:
  case OSSC_reduction:
  case OSSC_in:
  case OSSC_out:
  case OSSC_inout:
  case OSSC_concurrent:
  case OSSC_commutative:
  case OSSC_weakin:
  case OSSC_weakout:
  case OSSC_weakinout:
  case OSSC_weakconcurrent:
  case OSSC_weakcommutative:
  case OSSC_weakreduction:
  case OSSC_chunksize:
  case OSSC_grainsize:
  case OSSC_collapse:
    break;
  }
  llvm_unreachable("Invalid OmpSs simple clause kind");
}

const char *clang::getOmpSsSimpleClauseTypeName(OmpSsClauseKind Kind,
                                                unsigned Type) {
  switch (Kind) {
  case OSSC_default:
    switch (Type) {
    case OSSC_DEFAULT_unknown:
      return "unknown";
#define OMPSS_DEFAULT_KIND(Name)                                              \
  case OSSC_DEFAULT_##Name:                                                    \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'default' clause type");
  case OSSC_depend:
    switch (Type) {
    case OSSC_DEPEND_unknown:
      return "unknown";
#define OMPSS_DEPEND_KIND(Name)                                             \
  case OSSC_DEPEND_##Name:                                                   \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'depend' clause type");
  case OSSC_unknown:
  case OSSC_if:
  case OSSC_final:
  case OSSC_cost:
  case OSSC_priority:
  case OSSC_label:
  case OSSC_wait:
  case OSSC_onready:
  case OSSC_private:
  case OSSC_firstprivate:
  case OSSC_shared:
  case OSSC_reduction:
  case OSSC_in:
  case OSSC_out:
  case OSSC_inout:
  case OSSC_concurrent:
  case OSSC_commutative:
  case OSSC_weakin:
  case OSSC_weakout:
  case OSSC_weakinout:
  case OSSC_weakconcurrent:
  case OSSC_weakcommutative:
  case OSSC_weakreduction:
  case OSSC_chunksize:
  case OSSC_grainsize:
  case OSSC_collapse:
    break;
  }
  llvm_unreachable("Invalid OmpSs simple clause kind");
}

bool clang::isAllowedClauseForDirective(OmpSsDirectiveKind DKind,
                                        OmpSsClauseKind CKind) {
  assert(DKind <= OSSD_unknown);
  assert(CKind <= OSSC_unknown);
  switch (DKind) {
  case OSSD_task:
    switch (CKind) {
#define OMPSS_TASK_CLAUSE(Name)                                               \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_task_for:
    switch (CKind) {
#define OMPSS_TASK_FOR_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_taskloop:
    switch (CKind) {
#define OMPSS_TASKLOOP_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_taskloop_for:
    switch (CKind) {
#define OMPSS_TASKLOOP_FOR_CLAUSE(Name)                                       \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_declare_task:
    switch (CKind) {
#define OMPSS_DECLARE_TASK_CLAUSE(Name)                                       \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_taskwait:
    switch (CKind) {
#define OMPSS_TASKWAIT_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_release:
    switch (CKind) {
#define OMPSS_RELEASE_CLAUSE(Name)                                           \
  case OSSC_##Name:                                                            \
    return true;
#include "clang/Basic/OmpSsKinds.def"
    default:
      break;
    }
    break;
  case OSSD_declare_reduction:
  case OSSD_assert:
  case OSSD_unknown:
    break;
  }
  return false;
}

bool clang::isOmpSsPrivate(OmpSsClauseKind Kind) {
  return Kind == OSSC_private || Kind == OSSC_firstprivate;
}

bool clang::isOmpSsTaskingDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_task
    || Kind == OSSD_declare_task
    || isOmpSsLoopDirective(Kind);
}

bool clang::isOmpSsLoopDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_taskloop || Kind == OSSD_taskloop_for ||
         Kind == OSSD_task_for;
}

bool clang::isOmpSsTaskLoopDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_taskloop || Kind == OSSD_taskloop_for;
}

