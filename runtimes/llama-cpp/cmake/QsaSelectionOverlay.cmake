if(NOT TARGET llama)
    message(FATAL_ERROR "QSA selection requires the upstream llama target")
endif()
# Preparation composes QSA after file-backed rows in the same model/header tree.
# No target source relocation or second source list is necessary.
