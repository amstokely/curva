# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andy/CLionProjects/curva

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andy/CLionProjects/curva/build

# Utility rule file for pycurva_swig_compilation.

# Include any custom commands dependencies for this target.
include CMakeFiles/pycurva_swig_compilation.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pycurva_swig_compilation.dir/progress.make

CMakeFiles/pycurva_swig_compilation: CMakeFiles/pycurva.dir/pycurvaPYTHON.stamp

CMakeFiles/pycurva.dir/pycurvaPYTHON.stamp: /home/andy/CLionProjects/curva/swig/python/pycurva.i
CMakeFiles/pycurva.dir/pycurvaPYTHON.stamp: /home/andy/CLionProjects/curva/swig/python/pycurva.i
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/andy/CLionProjects/curva/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Swig compile /home/andy/CLionProjects/curva/swig/python/pycurva.i for python"
	/usr/bin/cmake -E make_directory /home/andy/CLionProjects/curva/build/CMakeFiles/pycurva.dir /home/andy/CLionProjects/curva/lib /home/andy/CLionProjects/curva/lib
	/usr/bin/cmake -E touch /home/andy/CLionProjects/curva/build/CMakeFiles/pycurva.dir/pycurvaPYTHON.stamp
	/usr/bin/cmake -E env SWIG_LIB=/home/andy/miniconda3/envs/curva/share/swig/4.0.2 /home/andy/miniconda3/envs/curva/bin/swig -python -outdir /home/andy/CLionProjects/curva/lib -c++ -interface _pycurva -I/usr/include -I/usr/local/cuda/include -I/home/andy/CLionProjects/curva/include/json/include -o /home/andy/CLionProjects/curva/lib/pycurvaPYTHON_wrap.cxx /home/andy/CLionProjects/curva/swig/python/pycurva.i

pycurva_swig_compilation: CMakeFiles/pycurva.dir/pycurvaPYTHON.stamp
pycurva_swig_compilation: CMakeFiles/pycurva_swig_compilation
pycurva_swig_compilation: CMakeFiles/pycurva_swig_compilation.dir/build.make
.PHONY : pycurva_swig_compilation

# Rule to build all files generated by this target.
CMakeFiles/pycurva_swig_compilation.dir/build: pycurva_swig_compilation
.PHONY : CMakeFiles/pycurva_swig_compilation.dir/build

CMakeFiles/pycurva_swig_compilation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pycurva_swig_compilation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pycurva_swig_compilation.dir/clean

CMakeFiles/pycurva_swig_compilation.dir/depend:
	cd /home/andy/CLionProjects/curva/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andy/CLionProjects/curva /home/andy/CLionProjects/curva /home/andy/CLionProjects/curva/build /home/andy/CLionProjects/curva/build /home/andy/CLionProjects/curva/build/CMakeFiles/pycurva_swig_compilation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pycurva_swig_compilation.dir/depend

