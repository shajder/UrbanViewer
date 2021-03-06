CMAKE_MINIMUM_REQUIRED (VERSION 2.8.12)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(COMMAND cmake_policy)
    # Works around warnings libraries linked against that don't
    # have absolute paths (e.g. -lpthreads)
    cmake_policy(SET CMP0003 NEW)

    # Works around warnings about escaped quotes in ADD_DEFINITIONS
    # statements.
    cmake_policy(SET CMP0005 NEW)

    # tell CMake to prefer CMake's own CMake modules when available
    # only available from cmake-2.8.4
    if(${CMAKE_MAJOR_VERSION} GREATER 2 OR
        (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} GREATER 8) OR
        (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} EQUAL 8 AND ${CMAKE_PATCH_VERSION} GREATER 3))
        cmake_policy(SET CMP0017 NEW)
    endif()

    # cmake-2.6.1 introduces policy cmp0008 decide how to treat full path libraries that do not appear to be valid library file names
    # quote from cvslog "Such libraries worked by accident in the VS IDE and Xcode generators in CMake 2.4 and below."
    if(${CMAKE_MAJOR_VERSION} GREATER 2 OR
        (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} GREATER 6) OR
        (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} EQUAL 6 AND ${CMAKE_PATCH_VERSION} GREATER 0))
        cmake_policy(SET CMP0008 OLD)
    endif()

    # disable autolinking to qtmain as we have our own main() functions (new in Qt 5.1)
    if(${CMAKE_MAJOR_VERSION} GREATER 2 OR
        (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} GREATER 8) OR
        (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} EQUAL 8 AND ${CMAKE_PATCH_VERSION} GREATER 10))
        cmake_policy(SET CMP0020 OLD)
    endif()
    # nicer version check - but needs at least CMake 2.6.2? Worth upgrading the requirements?
    #if("${CMAKE_VERSION}" VERSION_GREATER 2.8.10)
    # or even easier (available in cmake-2.6)
    #if(POLICY CMPxyzw)

endif()

PROJECT (UrbanViewer)
SET(OpenSceneGraph_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_MODULE_PATH "${OpenSceneGraph_SOURCE_DIR}/CMakeModules;${CMAKE_MODULE_PATH}")


MESSAGE(WARNING "${CMAKE_MODULE_PATH}")

INCLUDE(OsgMacroUtils)
INCLUDE(FindOSG)

SET(OPENTHREADS_SONAMES TRUE)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

SET(CMAKE_BUILD_TYPE "Release")
LIST(APPEND CMAKE_INCLUDE_PATH ${OSG_DIR}/include)
LIST(APPEND CMAKE_LIBRARY_PATH ${OSG_DIR}/lib)

IF("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
  IF(CMAKE_COMPILER_IS_GNUCXX)
    SET_PROPERTY(DIRECTORY APPEND PROPERTY  
      COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
    )
  ENDIF()
ENDIF()

CMAKE_MINIMUM_REQUIRED (VERSION 2.8.12)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)

MESSAGE(STATUS "CMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}")

FIND_PACKAGE(OpenSceneGraph REQUIRED osg osgDB osgUtil osgGA osgSim osgText osgViewer osgShadow)

INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OPENTHREADS_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSG_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGGA_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGDB_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGUTIL_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGQT_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGANIMATION_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGSHADOW_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSGTEXT_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${OSG_BINARY_DIR}/include ) 
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ./src ) 

SET(UrbanViewer_SOURCE_FILES 
  src/UrbanViewer.cpp
  src/PSSMLayered.cpp
  src/RenderingResources.cpp
)

LINK_DIRECTORIES (${LINK_DIRECTORIES} ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}) 
LINK_DIRECTORIES (${LINK_DIRECTORIES} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}) 
LINK_DIRECTORIES (${LINK_DIRECTORIES} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}) 

ADD_EXECUTABLE(UrbanViewer
  ${UrbanViewer_SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(UrbanViewer ${OPENTHREADS_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSG_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGDB_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGSHADOW_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGGA_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGUTIL_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGANIMATION_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGQT_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGVIEWER_LIBRARY} )
TARGET_LINK_LIBRARIES(UrbanViewer ${OSGTEXT_LIBRARY} )

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
ENDIF()
