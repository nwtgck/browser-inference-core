# Authorized experimental upstream changes

Use this directory name: `upstream-patches`.
For this browser/WebGPU image-runtime experiment the user has explicitly granted
agent discretion for upstream patches. Keep a reason, declared target paths, regression coverage and removal criteria.
The prepared-source receipt records the actual before/after hashes and patch
identities. Do not freeze full source files as a substitute for semantic tests;
selected upstream revisions are checked against the staged gitlinks. Do not edit checked-out submodules.
Do not extend this exception to `llama-cpp/`.
