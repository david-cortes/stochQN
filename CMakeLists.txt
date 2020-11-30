cmake_minimum_required (VERSION 3.9)
project (stochqn C)
set (stochqn_VERSION_MAJOR 0)
set (stochqn_VERSION_MINOR 1)
set(CMAKE_BUILD_TYPE Release)

# If you want to provide a custom BLAS library, put the full file path below,
# enclosed in double quotes, where it now says false.
# set(CUSTOM_BLAS "/path/to/mkl/libmkl_rt.so")
set(CUSTOM_BLAS false)

# If you want to compile for 'float' type, change this to USE_FLOAT
add_compile_definitions(USE_DOUBLE)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(BUILD_SHARED_LIBS True)
set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/stochqn.c)
add_library(stochqn SHARED ${SRC_FILES})

## This is taken from Armadillo: https://gitlab.com/conradsnicta/armadillo-code
if (NOT CUSTOM_BLAS)
	include(findBLAS/ARMA_FindMKL.cmake)
	include(findBLAS/ARMA_FindOpenBLAS.cmake)
	include(findBLAS/ARMA_FindATLAS.cmake)
	include(findBLAS/ARMA_FindBLAS.cmake)

	message(STATUS "     MKL_FOUND = ${MKL_FOUND}"     )
	message(STATUS "OpenBLAS_FOUND = ${OpenBLAS_FOUND}")
	message(STATUS "   ATLAS_FOUND = ${ATLAS_FOUND}"   )
	message(STATUS "    BLAS_FOUND = ${BLAS_FOUND}"    )
endif()


if (MKL_FOUND AND NOT CUSTOM_BLAS)
	set(BLAS_LIB ${MKL_LIBRARIES})
	set(HAS_MKL true)
elseif(OpenBLAS_FOUND AND NOT CUSTOM_BLAS)
	set(BLAS_LIB ${OpenBLAS_LIBRARIES})
	set(HAS_OPENBLAS true)
elseif (ATLAS_FOUND AND NOT CUSTOM_BLAS)
	set(BLAS_LIB ${ATLAS_LIBRARIES})
	set(HAS_ATLAS true)
elseif (BLAS_FOUND OR CUSTOM_BLAS)
	if (CUSTOM_BLAS)
		set(BLAS_LIB ${CUSTOM_BLAS})
	else()
		set(BLAS_LIB ${BLAS_LIBRARIES})
	endif()
	include(CheckLibraryExists)
    check_library_exists(${BLAS_LIB} "cblas_ddot" "" HAS_CBLAS)
    if (NOT HAS_CBLAS)
        check_library_exists(${BLAS_LIB} "ddot_" "" BLAS_UNDERSCORES)
        if (NOT BLAS_UNDERSCORES)
            check_library_exists(${BLAS_LIB} "ddot" "" BLAS_NO_UNDERSCORES)
            if (NOT BLAS_NO_UNDERSCORES)
                message( FATAL_ERROR "Cannot find any of {ddot, ddot_, cblas_ddot} in BLAS library.")
            endif()
        endif()
    endif()
	check_library_exists(${BLAS_LIB} "mkl_dcsrgemv" "" HAS_MKL)
    check_library_exists(${BLAS_LIB} "openblas_set_num_threads" "" HAS_OPENBLAS)
else()
    message( FATAL_ERROR "BLAS library not found." )
endif()
target_link_libraries(stochqn ${BLAS_LIB} )
IF(UNIX)
    target_link_libraries(stochqn m)
ENDIF(UNIX)

## https://stackoverflow.com/questions/12399422/how-to-set-linker-flags-for-openmp-in-cmakes-try-compile-function
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

configure_file(src/blasfuns.h.in blasfuns.h)
target_include_directories(stochqn PRIVATE include)
target_include_directories(stochqn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(stochqn PROPERTIES PUBLIC_HEADER include/stochqn.h)


if (MSVC)
    add_compile_options(/O2 /openmp)
else()
    add_compile_options(-O2 -fopenmp -march=native -std=c99)
endif()

include(GNUInstallDirs)

install(TARGETS stochqn
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_file(stochqn.pc.in stochqn.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/stochqn.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
