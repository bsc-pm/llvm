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

function warning()
{
  nice_message "WARNING" "$1"
}

function run()
{
  P="$@"
  echo "$P"
  "$@"
}

BUILDDIR=$(readlink -f $(pwd))
SRCDIR=$(dirname $(readlink -f $0))

################################################################################
# Detection of toolchain and sysroot
################################################################################

info "Using the current directory '${BUILDDIR}' as the build-dir"
info "Using '${SRCDIR}' as the source-dir"

if [ "$GCC_TOOLCHAIN" = "" ];
then
  die "Please, set the GCC_TOOLCHAIN environment variable to the location of the riscv64-unknown-linux-gnu toolchain. If you do not have a GCC toolchain yet, follow the instructions at https://pm.bsc.es/gitlab/EPI/project/wikis/install-toolchain"
fi

if [ ! -e "$GCC_TOOLCHAIN" ];
then
  die "GCC riscv64-unknown-linux-gnu toolchain location '$GCC_TOOLCHAIN' does not exist"
fi

if [ ! -d "$GCC_TOOLCHAIN" ];
then
  die "GCC riscv64-unknown-linux-gnu toolchain location '$GCC_TOOLCHAIN' must be a directory"
fi

info "Using GCC riscv64-unknown-linux-gnu toolchain at '$GCC_TOOLCHAIN'"

RISCV_SYSROOT=${RISCV_SYSROOT:-"${GCC_TOOLCHAIN}/sysroot"}
if [ ! -e "${RISCV_SYSROOT}" ];
then
  die "Linux RISC-V sysroot at '${RISCV_SYSROOT}' not found. You can override it setting the RISCV_SYSROOT environment variable"
fi

################################################################################
# Detection of build system
################################################################################

# We only support Makefiles or Ninja. Ninja is better as it is able of finer
# control

BUILD_SYSTEM="Unix Makefiles"
COMMAND_TO_BUILD="make -j$(nproc)"
NINJA_BIN=${NINJA_BIN:-$(which ninja)}

CMAKE_INVOCATION_EXTRA_FLAGS=""

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

################################################################################
# Detection of compiler
################################################################################

# We only support clang or gcc. While the compilers are similar in speed, clang
# allows using LLD which is noticeably faster than GNU ld

if [ -z "${COMPILER}" ];
then
  info "Automatic detection of compiler. Override the detection setting the COMPILER enviromment variable to either 'gcc' or 'clang', in which case CC and CXX will be used by cmake instead"
  if [ "$(which clang)" = "" ];
  then
    # Clang not found
    if [ "$(which gcc)" != "" ];
    then
      COMPILER="gcc"
      info "Using GCC $(gcc -dumpversion)"
      info "gcc: $(which gcc)"
      if [ "$(which g++)" != "" ];
      then
        info "g++: $(which g++)"
        # Sanity check
        if [ $(gcc -dumpversion) != $(g++ -dumpversion) ];
        then
          warning "gcc and g++ have different versions!"
        fi
        CC="$(which gcc)"
        CXX="$(which g++)"
        CMAKE_INVOCATION_EXTRA_FLAGS="-DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++)"
      else
        error "g++ not found in the PATH but gcc was found. This usually means that your system is missing development packages"
      fi
    fi
  else
    CLANG_VERSION=$(clang --version | head -n1 | sed 's/^.*version\s\+\([0-9]\+\(\.[0-9]\+\)\+\).*$/\1/')
    COMPILER="clang"
    info "Using clang ${CLANG_VERSION}"
    info "clang: $(which clang)"
    if [ "$(which clang++)" != "" ];
    then
      info "clang++: $(which clang++)"
      # Sanity check
      CLANGXX_VERSION=$(clang++ --version | head -n1 | sed 's/^.*version\s\+\([0-9]\+\(\.[0-9]\+\)\+\).*$/\1/')
      if [ "$CLANG_VERSION" != "$CLANGXX_VERSION" ];
      then
        warning "clang and clang++ have different versions!"
      fi
      CC="$(which clang)"
      CXX="$(which clang++)"
      CMAKE_INVOCATION_EXTRA_FLAGS="-DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++)"
    else
      error "clang++ not found in the PATH but clang was found. You may have to review your installation"
    fi
  fi
elif [ "${COMPILER}" = gcc ];
then
  CC=${CC:-"$(which gcc)"}
  CXX=${CXX:-"$(which g++)"}
elif [ "${COMPILER}" = clang ];
then
  CC=${CC:-"$(which clang)"}
  CXX=${CXX:-"$(which clang++)"}
fi

################################################################################
# Detection of the linker
################################################################################

# If using clang we try to use lld

LINKER=gnu-ld
if [ "$COMPILER" = "clang" ];
then
 if ( ${CC} -fuse-ld=lld -Wl,--version 2> /dev/null ) | grep -q "^LLD";
 then
   info "Using LLD"
   LINKER=lld
 else
   info "Using GNU ld because we didn't find lld"
 fi
else
  info "Using GNU ld because we are using gcc"
fi

if [ "$LINKER" = "lld" ];
then
  CMAKE_INVOCATION_EXTRA_FLAGS="${CMAKE_INVOCATION_EXTRA_FLAGS} -DLLVM_ENABLE_LLD=ON"
fi

if [ "$BUILD_SYSTEM" = "Ninja" ];
then
  info "Limiting concurrent linking jobs to 1"
  CMAKE_INVOCATION_EXTRA_FLAGS="${CMAKE_INVOCATION_EXTRA_FLAGS} -DLLVM_PARALLEL_LINK_JOBS=1"
else
  warning "Makefiles do not allow limiting the concurrent linking jobs"
fi

################################################################################
# cmake
################################################################################

info "Running cmake..."
run cmake -G "${BUILD_SYSTEM}" ${SRCDIR} \
   -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=RISCV \
   -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
   -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
   -DDEFAULT_SYSROOT=${RISCV_SYSROOT} \
   -DGCC_INSTALL_PREFIX=${GCC_TOOLCHAIN} \
   ${CMAKE_INVOCATION_EXTRA_FLAGS}

if [ $? = 0 ];
then
  echo ""
  echo "cmake finished successfully, you may want to tune the configuration in CMakeCache.txt or using a GUI tool like ccmake"
  echo ""
  echo "Now run '${COMMAND_TO_BUILD}' to build."
fi
