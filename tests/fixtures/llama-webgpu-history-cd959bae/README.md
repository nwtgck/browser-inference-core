# Historical WebGPU composer input

This deliberately frozen, source-only fixture belongs to tests of the historical
stage composer and its byte guards. It is **not** the input of current native
compatibility tests or the production build. The archive includes the upstream
MIT license. `manifest.json` records every retained file and the source of the
capture; it does not assert possession of upstream Git history.

Do not refresh this fixture simply because the live submodule changes. A new
historical baseline needs a separately reviewed fixture and comparison. Current
behavior is checked against `LCB_TEST_LLAMA_SOURCE` by the production source-plan,
complete-header, CMake, numerical, and browser tests.

Migration is incremental: `source/test_historical_guards.py` and the historical
aligned-word regression own this fixture. Current standalone/provider checks
use separate live-input classes.
Other stage-specific legacy tests still need their input and semantic contracts
classified; this fixture is not silently substituted for their live source.

Accepted patch files are frozen under `patches/`, with the bicore license.
Historical guard tests explicitly pass that directory rather than silently
reading production patches. The frozen input is never installed as a vendor.

The snapshot/scheduler historical comparison additionally owns the CPU registry
source. Its bytes were checked against the existing snapshot contract before
adding it; all previously captured entries remain byte-for-byte unchanged.
Current copy behavior uses `test_current_copy_recording`, not this archive.
