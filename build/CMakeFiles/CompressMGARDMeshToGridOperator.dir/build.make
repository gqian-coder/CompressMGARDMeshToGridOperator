# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_SOURCE_DIR = /home/qian/Software/CompressMGARDMeshToGridOperator

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qian/Software/CompressMGARDMeshToGridOperator/build

# Include any dependencies generated for this target.
include CMakeFiles/CompressMGARDMeshToGridOperator.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CompressMGARDMeshToGridOperator.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CompressMGARDMeshToGridOperator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CompressMGARDMeshToGridOperator.dir/flags.make

CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o: CMakeFiles/CompressMGARDMeshToGridOperator.dir/flags.make
CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o: ../CompressMGARDMeshToGridOperator.cpp
CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o: CMakeFiles/CompressMGARDMeshToGridOperator.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qian/Software/CompressMGARDMeshToGridOperator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o -MF CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o.d -o CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o -c /home/qian/Software/CompressMGARDMeshToGridOperator/CompressMGARDMeshToGridOperator.cpp

CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qian/Software/CompressMGARDMeshToGridOperator/CompressMGARDMeshToGridOperator.cpp > CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.i

CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qian/Software/CompressMGARDMeshToGridOperator/CompressMGARDMeshToGridOperator.cpp -o CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.s

# Object files for target CompressMGARDMeshToGridOperator
CompressMGARDMeshToGridOperator_OBJECTS = \
"CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o"

# External object files for target CompressMGARDMeshToGridOperator
CompressMGARDMeshToGridOperator_EXTERNAL_OBJECTS =

libCompressMGARDMeshToGridOperator.so: CMakeFiles/CompressMGARDMeshToGridOperator.dir/CompressMGARDMeshToGridOperator.cpp.o
libCompressMGARDMeshToGridOperator.so: CMakeFiles/CompressMGARDMeshToGridOperator.dir/build.make
libCompressMGARDMeshToGridOperator.so: /home/qian/Software/ADIOS2/install-adios/lib/libadios2_cxx11.so.2.10.0
libCompressMGARDMeshToGridOperator.so: /home/qian/Software/ADIOS2/install-adios/lib/libadios2_core.so.2.10.0
libCompressMGARDMeshToGridOperator.so: /home/qian/Software/MGARD/install-cuda-turing/lib/libmgard.so.1.5.2
libCompressMGARDMeshToGridOperator.so: /home/qian/Software/MGARD/install-cuda-turing/lib/libprotobuf.so
libCompressMGARDMeshToGridOperator.so: /usr/lib/x86_64-linux-gnu/libz.so
libCompressMGARDMeshToGridOperator.so: /home/qian/Software/MGARD/install-cuda-turing/lib/libzstd.so
libCompressMGARDMeshToGridOperator.so: /home/qian/Software/MGARD/install-cuda-turing/lib/libnvcomp.so
libCompressMGARDMeshToGridOperator.so: /usr/local/cuda/lib64/libcudart.so
libCompressMGARDMeshToGridOperator.so: CMakeFiles/CompressMGARDMeshToGridOperator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qian/Software/CompressMGARDMeshToGridOperator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libCompressMGARDMeshToGridOperator.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CompressMGARDMeshToGridOperator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CompressMGARDMeshToGridOperator.dir/build: libCompressMGARDMeshToGridOperator.so
.PHONY : CMakeFiles/CompressMGARDMeshToGridOperator.dir/build

CMakeFiles/CompressMGARDMeshToGridOperator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CompressMGARDMeshToGridOperator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CompressMGARDMeshToGridOperator.dir/clean

CMakeFiles/CompressMGARDMeshToGridOperator.dir/depend:
	cd /home/qian/Software/CompressMGARDMeshToGridOperator/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qian/Software/CompressMGARDMeshToGridOperator /home/qian/Software/CompressMGARDMeshToGridOperator /home/qian/Software/CompressMGARDMeshToGridOperator/build /home/qian/Software/CompressMGARDMeshToGridOperator/build /home/qian/Software/CompressMGARDMeshToGridOperator/build/CMakeFiles/CompressMGARDMeshToGridOperator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CompressMGARDMeshToGridOperator.dir/depend

