#!/bin/bash
# This is a convenience script to set up cmake in the EPI project.

function nice_message()
{
  local prefix="$1"
  local message="$2"

  if [ "$(which fmt)" = "" ];
  then
    # Best effort if no fmt available in the PATH
    echo "$prefix: $message"
  else
    echo "$prefix: $message" | fmt -w80 --prefix="$prefix: "
  fi
}

function die()
{
  nice_message "ERROR" "$1"
  exit 1
}

function info()
{
  nice_message "INFO" "$1"
}

BUILDDIR=$(readlink -f $(pwd))
SRCDIR=$(dirname $(readlink -f $0))

info "Using the current directory '${BUILDDIR}' as the build-dir"
info "Using '${SRCDIR}' as the source-dir"

if [ "$GCC_TOOLCHAIN" = "" ];
then
  die "Please, set the GCC_TOOLCHAIN environment variable to the location of the riscv64-unknown-linux-gnu toolchain. If you do not have a GCC toolchain yet, follow the instructions at https://pm.bsc.es/gitlab/EPI/project/wikis/install-toolchain"
fi

if [ ! -e "$GCC_TOOLCHAIN" ];
then
  die "GCC toolchain location '$GCC_TOOLCHAIN' does not exist"
fi

if [ ! -d "$GCC_TOOLCHAIN" ];
then
  die "GCC toolchain location '$GCC_TOOLCHAIN' must be a directory"
fi

info "Using GCC toolchain at '$GCC_TOOLCHAIN'"

RISCV_SYSROOT=${RISCV_SYSROOT:-"${GCC_TOOLCHAIN}/sysroot"}
if [ ! -e "${RISCV_SYSROOT}" ];
then
  die "Linux sysroot at '${RISCV_SYSROOT}' not found. You can override it setting the RISCV_SYSROOT environment variable"
fi

BUILD_SYSTEM="Unix Makefiles"
COMMAND_TO_BUILD="make -j$(nproc)"
NINJA_BIN=${NINJA_BIN:-$(which ninja)}

if [ "${NINJA_BIN}" = "" ];
then
  info "Using Makefiles as build system because 'ninja' wasn't found in your PATH. You can override the location setting the NINJA_BIN environment variable"
elif [ ! -x "${NINJA_BIN}" ];
then
  info "Using Makefiles as build system because '${NINJA_BIN}' is not executable. You can override the location setting the NINJA_BIN environment variable"
else
  info "Using ninja in '${NINJA_BIN}'"
  BUILD_SYSTEM="Ninja"
  COMMAND_TO_BUILD="ninja"
  # Do not presume we can use 'ninja' as if it were in the path
  if [ "$(which ninja)" = "" ];
  then
    COMMAND_TO_BUILD="${NINJA_BIN}"
  fi
fi

info "Running cmake..."
set -x
cmake -G "${BUILD_SYSTEM}" ${SRCDIR} \
   -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=RISCV \
   -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
   -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
   -DDEFAULT_SYSROOT=${RISCV_SYSROOT} \
   -DGCC_INSTALL_PREFIX=${GCC_TOOLCHAIN}
set +x

if [ $? = 0 ];
then
  echo ""
  echo "cmake finished successfully, you may want to tune the configuration in CMakeCache.txt or using a GUI tool like ccmake"
  echo ""
  echo "Now run '${COMMAND_TO_BUILD}' to build."
fi
