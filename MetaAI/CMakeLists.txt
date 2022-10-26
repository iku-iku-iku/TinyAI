project(MetaAI)

set(CMAKE_CXX_STANDARD 20)

set(HEADER_FILES Components.h Util.h)

add_library(MetaAILib STATIC ${HEADER_FILES})
set_target_properties(MetaAILib PROPERTIES LINKER_LANGUAGE CXX)