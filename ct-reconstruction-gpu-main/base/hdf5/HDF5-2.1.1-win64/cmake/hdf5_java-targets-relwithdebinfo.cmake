#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdf5::hdf5_java" for configuration "RelWithDebInfo"
set_property(TARGET hdf5::hdf5_java APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(hdf5::hdf5_java PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/hdf5_java.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/bin/hdf5_java.dll"
  )

list(APPEND _cmake_import_check_targets hdf5::hdf5_java )
list(APPEND _cmake_import_check_files_for_hdf5::hdf5_java "${_IMPORT_PREFIX}/lib/hdf5_java.lib" "${_IMPORT_PREFIX}/bin/hdf5_java.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
