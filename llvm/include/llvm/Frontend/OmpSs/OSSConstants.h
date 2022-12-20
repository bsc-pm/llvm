//===- OSSConstants.h - OmpSs related constants and helpers ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines constans and helpers used when dealing with OmpSs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OMPSS_OSSCONSTANTS_H
#define LLVM_FRONTEND_OMPSS_OSSCONSTANTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OmpSs/OSS.h.inc"

namespace llvm {
namespace oss {

/// IDs for the different default kinds.
enum class DefaultKind {
#define OSS_DEFAULT_KIND(Enum, Str) Enum,
#include "llvm/Frontend/OmpSs/OSSKinds.def"
};

#define OSS_DEFAULT_KIND(Enum, ...)                                            \
  constexpr auto Enum = oss::DefaultKind::Enum;
#include "llvm/Frontend/OmpSs/OSSKinds.def"

} // end namespace oss

} // end namespace llvm

#endif // LLVM_FRONTEND_OMPSS_OSSCONSTANTS_H
