//===--- OmpSsKinds.cpp - Token Kinds Support ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
using namespace llvm::oss;

// OmpSsDirectiveKind clang::getOmpSsDirectiveKind(StringRef Str) {
//   return llvm::StringSwitch<OmpSsDirectiveKind>(Str)
// #define OMPSS_DIRECTIVE(Name) .Case(#Name, OSSD_##Name)
// #define OMPSS_DIRECTIVE_EXT(Name, Str) .Case(Str, OSSD_##Name)
// #include "clang/Basic/OmpSsKinds.def"
//       .Default(OSSD_unknown);
// }

// const char *clang::getOmpSsDirectiveName(OmpSsDirectiveKind Kind) {
//   assert(Kind <= OSSD_unknown);
//   switch (Kind) {
//   case OSSD_unknown:
//     return "unknown";
// #define OMPSS_DIRECTIVE(Name)                                                 \
//   case OSSD_##Name:                                                            \
//     return #Name;
// #define OMPSS_DIRECTIVE_EXT(Name, Str)                                        \
//   case OSSD_##Name:                                                            \
//     return Str;
// #include "clang/Basic/OmpSsKinds.def"
//     break;
//   }
//   llvm_unreachable("Invalid OmpSs directive kind");
// }

// OmpSsClauseKind clang::getOmpSsClauseKind(StringRef Str) {
//   return llvm::StringSwitch<OmpSsClauseKind>(Str)
// #define OMPSS_CLAUSE(Name, Class) .Case(#Name, OSSC_##Name)
// #define OMPSS_CLAUSE_ALIAS(Alias, Name) .Case(#Alias, OSSC_##Alias)
// #include "clang/Basic/OmpSsKinds.def"
//       .Default(OSSC_unknown);
// }

// const char *clang::getOmpSsClauseName(OmpSsClauseKind Kind) {
//   assert(Kind <= OSSC_unknown);
//   switch (Kind) {
//   case OSSC_unknown:
//     return "unknown";
// #define OMPSS_CLAUSE(Name, Class)                                             \
//   case OSSC_##Name:                                                            \
//     return #Name;
// #define OMPSS_CLAUSE_ALIAS(Alias, Name)                                       \
//   case OSSC_##Alias:                                                           \
//     return #Alias;
// #include "clang/Basic/OmpSsKinds.def"
//   }
//   llvm_unreachable("Invalid OmpSs clause kind");
// }

unsigned clang::getOmpSsSimpleClauseType(OmpSsClauseKind Kind,
                                         StringRef Str) {
  switch (Kind) {
  case OSSC_default:
    return llvm::StringSwitch<unsigned>(Str)
#define OSS_DEFAULT_KIND(Enum, Name) .Case(Name, unsigned(Enum))
#include "llvm/Frontend/OmpSs/OSSKinds.def"
        .Default(unsigned(llvm::oss::OSS_DEFAULT_unknown));
  case OSSC_depend:
    return llvm::StringSwitch<OmpSsDependClauseKind>(Str)
#define OMPSS_DEPEND_KIND(Name) .Case(#Name, OSSC_DEPEND_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEPEND_unknown);
  case OSSC_device:
    return llvm::StringSwitch<OmpSsDeviceClauseKind>(Str)
#define OMPSS_DEVICE_KIND(Name) .Case(#Name, OSSC_DEVICE_##Name)
#include "clang/Basic/OmpSsKinds.def"
        .Default(OSSC_DEVICE_unknown);
  case OSSC_unknown:
  case OSSC_immediate:
  case OSSC_microtask:
  case OSSC_if:
  case OSSC_final:
  case OSSC_cost:
  case OSSC_priority:
  case OSSC_label:
  case OSSC_wait:
  case OSSC_update:
  case OSSC_shmem:
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
  case OSSC_on:
  case OSSC_weakin:
  case OSSC_weakout:
  case OSSC_weakinout:
  case OSSC_weakconcurrent:
  case OSSC_weakcommutative:
  case OSSC_weakreduction:
  case OSSC_chunksize:
  case OSSC_grainsize:
  case OSSC_unroll:
  case OSSC_collapse:
  case OSSC_ndrange:
  case OSSC_read:
  case OSSC_write:
  case OSSC_capture:
  case OSSC_compare:
  case OSSC_seq_cst:
  case OSSC_acq_rel:
  case OSSC_acquire:
  case OSSC_release:
  case OSSC_relaxed:
    break;
  }
  llvm_unreachable("Invalid OmpSs simple clause kind");
}

const char *clang::getOmpSsSimpleClauseTypeName(OmpSsClauseKind Kind,
                                                unsigned Type) {
  switch (Kind) {
  case OSSC_default:
    switch (llvm::oss::DefaultKind(Type)) {
#define OSS_DEFAULT_KIND(Enum, Name)                                           \
  case Enum:                                                                   \
    return Name;
#include "llvm/Frontend/OmpSs/OSSKinds.def"
    }
    llvm_unreachable("Invalid OmpSs-2 'default' clause type");
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
  case OSSC_device:
    switch (Type) {
    case OSSC_DEVICE_unknown:
      return "unknown";
#define OMPSS_DEVICE_KIND(Name)                                             \
  case OSSC_DEVICE_##Name:                                                   \
    return #Name;
#include "clang/Basic/OmpSsKinds.def"
    }
    llvm_unreachable("Invalid OmpSs 'device' clause type");
  case OSSC_unknown:
  case OSSC_immediate:
  case OSSC_microtask:
  case OSSC_if:
  case OSSC_final:
  case OSSC_cost:
  case OSSC_priority:
  case OSSC_label:
  case OSSC_wait:
  case OSSC_update:
  case OSSC_shmem:
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
  case OSSC_on:
  case OSSC_weakin:
  case OSSC_weakout:
  case OSSC_weakinout:
  case OSSC_weakconcurrent:
  case OSSC_weakcommutative:
  case OSSC_weakreduction:
  case OSSC_chunksize:
  case OSSC_grainsize:
  case OSSC_unroll:
  case OSSC_collapse:
  case OSSC_ndrange:
  case OSSC_read:
  case OSSC_write:
  case OSSC_capture:
  case OSSC_compare:
  case OSSC_seq_cst:
  case OSSC_acq_rel:
  case OSSC_acquire:
  case OSSC_release:
  case OSSC_relaxed:
    break;
  }
  llvm_unreachable("Invalid OmpSs simple clause kind");
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
  return getDirectiveAssociation(Kind) == Association::Loop;
}

bool clang::isOmpSsTaskLoopDirective(OmpSsDirectiveKind Kind) {
  return Kind == OSSD_taskloop || Kind == OSSD_taskloop_for;
}

