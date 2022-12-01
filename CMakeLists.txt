cmake_minimum_required(VERSION 3.21)
project(curva LANGUAGES C CXX CUDA)
set(BUILD_PYTHON ON CACHE BOOL "")
set(BUILD_TCL ON CACHE BOOL "")
set(BUILD_TESTS ON CACHE BOOL "")
set(CONDA_DIR "" CACHE STRING "")
find_package (Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Doxygen REQUIRED)



set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(SRC_DIR ${ROOT_DIR}/src)
set(CUDA_SRC_DIR ${SRC_DIR}/cuda)
set(INCLUDE_DIR ${ROOT_DIR}/include)
set(LIB_DIR ${ROOT_DIR}/lib)
set(PYCURVA_DIR ${ROOT_DIR}/wrappers/pycurva/pycurva)
set(TCLCURVA_DIR ${ROOT_DIR}/wrappers/tclcurva)
set(TEST_DIR ${ROOT_DIR}/tests)
set(SWIG_FILES_DIR ${ROOT_DIR}/swig)
set(TCL_SWIG_FILES_DIR ${SWIG_FILES_DIR}/tcl)
set(TCL_SWIG_INTERFACE_FILE ${TCL_SWIG_FILES_DIR}/tclcurva.i)
set(PYTHON_SWIG_FILES_DIR ${SWIG_FILES_DIR}/python)
set(PYTHON_SWIG_INTERFACE_FILE ${PYTHON_SWIG_FILES_DIR}/pycurva.i)
set(TCL_INCLUDE_DIR ${CONDA_DIR}/include)
set(PYTHON_LIBRARY ${CONDA_DIR}/bin)

IF(NOT EXISTS ${LIB_DIR})
    install(DIRECTORY DESTINATION ${LIB_DIR})
ENDIF()
IF(NOT EXISTS ${TCLCURVA_DIR})
    install(DIRECTORY DESTINATION ${TCLCURVA_DIR})
ENDIF()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TEST_DIR})
find_package(boost_filesystem REQUIRED CONFIG)

find_package(Boost REQUIRED COMPONENTS SYSTEM FILESYSTEM)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${ROOT_DIR}/include/json/include")
set(ZLIB_INCLUDE_DIR ${CONDA_DIR}/include)
list(APPEND PYCURVA_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
list(APPEND PYCURVA_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})
list(APPEND PYCURVA_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR})
list(APPEND PYCURVA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND PYCURVA_INCLUDE_DIRS ${ROOT_DIR}/include/json/include)
list(APPEND TCL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND TCL_INCLUDE_DIRS ${TCL_INCLUDE_DIR})
list(APPEND TCL_INCLUDE_DIRS ${ROOT_DIR}/include/json/include)



find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
set_source_files_properties(${TCL_SWIG_INTERFACE_FILE} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${PYTHON_SWIG_INTERFACE_FILE} PROPERTIES CPLUSPLUS ON)

add_library(curva SHARED
        ${SRC_DIR}/curva.cpp
        ${SRC_DIR}/calculation.cpp
        ${SRC_DIR}/serializer.cpp
        ${SRC_DIR}/matrix/curva_vector.cpp
        ${SRC_DIR}/matrix/curva_matrix.cu
        ${SRC_DIR}/utils.cpp
        ${SRC_DIR}/dcd/dcd.cpp
        ${SRC_DIR}/cnpy/cnpy.cpp
        ${CUDA_SRC_DIR}/cudautils.cpp
        ${CUDA_SRC_DIR}/mutual_information.cu
        ${CUDA_SRC_DIR}/cutoff_matrix.cu
        ${SRC_DIR}/atom.cpp
        ${SRC_DIR}/coordinates.cpp
        ${SRC_DIR}/node.cpp
        ${SRC_DIR}/nodes.cpp
        ${SRC_DIR}/utils.cpp
        ${SRC_DIR}/atoms.cpp
        )
target_compile_options(curva PRIVATE -O3)
set_target_properties(curva PROPERTIES
        OUTPUT_DIRECTORY ${LIB_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61
        )
target_link_libraries(curva ${Boost_FILESYSTEM_LIBRARY_RELEASE})
target_link_libraries(curva ${Boost_SYSTEM_LIBRARY_RELEASE})

set(DOXYGEN_PROJECT_NAME CuRva)
set(DOXYGEN_PREDEFINED DOXYGEN_IGNORE)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_GENERATE_DOCBOOK YES)
set(DOXYGEN_GENERATE_LATEX YES)
set(DOXYGEN_USE_PDFLATEX YES)
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_ALIASES
        [[span{1}="<A>\1</A>"]]
        [[bold{1}="<b>\1</b>"]]
        [[Example{1}="\bold{Example}\code{.cpp}\1\endcode"]]
        [[Output{1}="Output:\code{.unparsed}\1\endcode"]]
        )
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/docs)
set(DOXYGEN_EXAMPLE_PATH ${CMAKE_SOURCE_DIR}/include/atom.h)
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/atom.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/atoms.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/calculation.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/coordinates.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/curva.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/element_properties.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/node.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/nodes.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/serializer.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/utils.h )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/matrix )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/custom_iterators )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/cuda )
list(APPEND CURVA_DOCUMENTATION_INPUT_FILES ${CMAKE_SOURCE_DIR}/include/chemical_properties )
doxygen_add_docs(
       curva_docs
        ${CURVA_DOCUMENTATION_INPUT_FILES}
        )
set(DOCUMENTATION_ARGS "Doxyfile.curva_docs")
add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOCUMENTATION_ARGS}
        )



if (${BUILD_TCL})
    swig_add_module(tclcurva tcl ${TCL_SWIG_INTERFACE_FILE}
            ${SRC_DIR}/curva.cpp
            ${SRC_DIR}/calculation.cpp
            ${SRC_DIR}/utils.cpp
            ${SRC_DIR}/atom.cpp
            ${SRC_DIR}/matrix/curva_matrix.cu
            ${SRC_DIR}/matrix/curva_vector.cpp
            ${SRC_DIR}/utils.cpp
            ${SRC_DIR}/node.cpp
            ${SRC_DIR}/nodes.cpp
            ${SRC_DIR}/atoms.cpp
            ${SRC_DIR}/coordinates.cpp
            ${SRC_DIR}/cnpy/cnpy.cpp
            OUTPUT_DIR ${TCLCURVA_DIR})
    target_compile_options(tclcurva PRIVATE -O3)
    set_target_properties(
            tclcurva PROPERTIES
            CUDA_ARCHITECTURES 61
            CUDA_SEPARABLE_COMPILATION ON
            INCLUDE_DIRECTORIES "${TCL_INCLUDE_DIRS}"
            LIBRARY_OUTPUT_DIRECTORY ${TCLCURVA_DIR}
    )
    swig_link_libraries(tclcurva curva)
endif ()

if (${BUILD_PYTHON})
    swig_add_module(pycurva python ${PYTHON_SWIG_INTERFACE_FILE}
            ${SRC_DIR}/curva.cpp
            ${SRC_DIR}/calculation.cpp
            ${SRC_DIR}/matrix/curva_matrix.cu
            ${SRC_DIR}/matrix/curva_vector.cpp
            ${SRC_DIR}/atom.cpp
            ${SRC_DIR}/utils.cpp
            ${SRC_DIR}/atoms.cpp
            ${SRC_DIR}/node.cpp
            ${SRC_DIR}/nodes.cpp
            ${SRC_DIR}/coordinates.cpp
            ${SRC_DIR}/cnpy/cnpy.cpp
            OUTPUT_DIR ${PYCURVA_DIR})
    target_compile_options(pycurva PRIVATE -O3)
    set_target_properties(
            pycurva PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 61
            INCLUDE_DIRECTORIES "${PYCURVA_INCLUDE_DIRS}"
            LIBRARY_OUTPUT_DIRECTORY ${PYCURVA_DIR}
    )
    swig_link_libraries(pycurva ${CONDA_DIR}/lib/libz.so)
    swig_link_libraries(pycurva curva)
    set(PYTHON_BUILD_EXT_ARGS "setup.py" "build_ext" "--inplace")
    set(PYTHON_BUILD_ARGS "setup.py" "build")
    set(PYTHON_INSTALL_ARGS "setup.py" "install")
    add_custom_target(PythonInstall
            COMMAND ${Python3_EXECUTABLE} ${PYTHON_BUILD_EXT_ARGS}
            COMMAND ${Python3_EXECUTABLE} ${PYTHON_BUILD_ARGS}
            COMMAND ${Python3_EXECUTABLE} ${PYTHON_INSTALL_ARGS}
            WORKING_DIRECTORY ../wrappers/pycurva)
endif ()

if (${BUILD_TESTS})
    find_package(ZLIB REQUIRED)
    add_executable(gpu_generalized_correlation_single_window_test
            ${TEST_DIR}/gpu_generalized_correlation_single_window_test.cpp
            ${SRC_DIR}/curva.cpp
            ${SRC_DIR}/serializer.cpp
            ${SRC_DIR}/calculation.cpp
            ${SRC_DIR}/utils.cpp
            ${SRC_DIR}/matrix/curva_matrix.cu
            ${SRC_DIR}/matrix/curva_vector.cpp
            ${SRC_DIR}/dcd/dcd.cpp
            ${SRC_DIR}/node.cpp
            ${SRC_DIR}/nodes.cpp
            ${SRC_DIR}/cnpy/cnpy.cpp
            ${CUDA_SRC_DIR}/mutual_information.cu
            ${CUDA_SRC_DIR}/cutoff_matrix.cu
            ${CUDA_SRC_DIR}/cudautils.cpp
            ${SRC_DIR}/atom.cpp
            ${SRC_DIR}/atoms.cpp
            ${SRC_DIR}/coordinates.cpp
            )
    set_target_properties(
            gpu_generalized_correlation_single_window_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 61
    )
    target_compile_options(
            gpu_generalized_correlation_single_window_test PRIVATE -O3)
    target_link_libraries(gpu_generalized_correlation_single_window_test ${ZLIB_LIBRARIES})
    target_link_libraries(gpu_generalized_correlation_single_window_test ${Boost_FILESYSTEM_LIBRARY_RELEASE})
    target_link_libraries(gpu_generalized_correlation_single_window_test ${Boost_SYSTEM_LIBRARY_RELEASE})
endif ()
