# Shared by the real build and configure-only overlay tests. Source identity must
# be selected before invoking Python or resolving a vendor/patch-plan path.
get_filename_component(LCB_SOURCE_CONFIG_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LCB_SOURCE_REGISTRY "${LCB_SOURCE_CONFIG_ROOT}/config/sources.json")
file(READ "${LCB_SOURCE_REGISTRY}" LCB_SOURCE_REGISTRY_JSON)
string(JSON LCB_DEFAULT_SOURCE GET "${LCB_SOURCE_REGISTRY_JSON}" defaultSource)
set(LCB_SOURCE_ID "${LCB_DEFAULT_SOURCE}" CACHE STRING "Configured upstream source identity")
string(JSON LCB_SELECTED_SOURCE ERROR_VARIABLE LCB_SOURCE_ERROR
    GET "${LCB_SOURCE_REGISTRY_JSON}" sources "${LCB_SOURCE_ID}")
if(LCB_SOURCE_ERROR)
    message(FATAL_ERROR "Unknown LCB_SOURCE_ID '${LCB_SOURCE_ID}' in ${LCB_SOURCE_REGISTRY}")
endif()
string(JSON LCB_VENDOR_PATH GET "${LCB_SELECTED_SOURCE}" vendorPath)
string(JSON LCB_PIN_FILE GET "${LCB_SELECTED_SOURCE}" pinFile)
string(JSON LCB_PATCH_SERIES_PATH GET "${LCB_SELECTED_SOURCE}" patchSeries)
# Keep an explicit source override for native fixtures. Do not cache the derived
# default: changing SOURCE_ID on reconfigure must not keep the old vendor path.
if(NOT DEFINED LCB_LLAMA_SOURCE)
    set(LCB_LLAMA_SOURCE "${LCB_SOURCE_CONFIG_ROOT}/${LCB_VENDOR_PATH}")
elseif(LCB_LLAMA_SOURCE STREQUAL "")
    message(FATAL_ERROR "LCB_LLAMA_SOURCE must not be empty")
endif()
set(LCB_PATCH_SERIES_FILE "${LCB_SOURCE_CONFIG_ROOT}/${LCB_PATCH_SERIES_PATH}")
file(READ "${LCB_PATCH_SERIES_FILE}" LCB_PATCH_SERIES_JSON)
string(JSON LCB_PATCH_COUNT LENGTH "${LCB_PATCH_SERIES_JSON}" patches)
set(LCB_PATCH_INPUTS "")
if(LCB_PATCH_COUNT GREATER 0)
    math(EXPR LCB_PATCH_LAST "${LCB_PATCH_COUNT} - 1")
    foreach(LCB_PATCH_INDEX RANGE ${LCB_PATCH_LAST})
        string(JSON LCB_PATCH_FILE GET "${LCB_PATCH_SERIES_JSON}" patches ${LCB_PATCH_INDEX} file)
        list(APPEND LCB_PATCH_INPUTS "${LCB_SOURCE_CONFIG_ROOT}/${LCB_PATCH_FILE}")
    endforeach()
endif()
# Track the registered paths, not a source-ID-derived directory convention.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_REGISTRY}"
    "${LCB_SOURCE_CONFIG_ROOT}/${LCB_PIN_FILE}"
    "${LCB_PATCH_SERIES_FILE}"
    "${LCB_SOURCE_CONFIG_ROOT}/scripts/source_config.py"
    "${LCB_SOURCE_CONFIG_ROOT}/scripts/source_git.py"
    ${LCB_PATCH_INPUTS})
