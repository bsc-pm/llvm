//===-- Wrapper for C standard stdio.h declarations on the GPU ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include_next <stdio.h>

#ifndef __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__
#define __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__

#if !defined(_OPENMP) && !defined(__HIP__) && !defined(__CUDA__)
#error "This file is for GPU offloading compilation only"
#endif

#if __has_include(<llvm-libc-decls/stdio.h>)

#if defined(__HIP__) || defined(__CUDA__)
#define __LIBC_ATTRS __attribute__((device))
#endif

#pragma omp begin declare target

#include <llvm-libc-decls/stdio.h>

#pragma omp end declare target

#undef __LIBC_ATTRS

#endif

#endif // __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__
