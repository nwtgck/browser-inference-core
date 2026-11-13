# Profile query budget regression

The profile QuerySet has 4096 entries and every timestamped dispatch consumes two.
A graph checks the nominal submission threshold only after encoding an entire op.
Reserving queries for that nominal size misses multi-dispatch overshoot.

The pinned failing predecessor is executed through actual extracted graph, build,
arena and cache helpers. A CPU Encode callback supplies source-derived dispatch
counts; it is not a full GGML graph allocator/factory or GPU execution. The
collector verifies names/query counts and drains the deferred CPU queue model.

An argsort width of `1024*W` has 11 dispatches; `128*W` has 8, where `W` is the real
factory's workgroup size. The 187-node sequence `11` repeated 185 times, then `8`,
then `11`, uses at most 74 arena slots but attempts query 4097 before the graph can
collect. It is minimal under an 11-dispatch-per-op bound. A second 518-node
example uses only 4/3-dispatch ops and much smaller widths. Both predecessors must
fail, while the prepared helper must preserve every observed parameter value.
Tests include 2048/2049 dispatches, maximum 74 slots, the existing 75-slot arena
rejection, and profile initialization capacity 74/2048/2049.

If the arena capacity is `S <= 4096/2`, a fresh QuerySet fits any accepted batch.
After a boundary where collection was skipped, `q+2*S<4096`; the next batch can
consume at most `S` slots, so every timestamp fits. The initial-capacity assertion
keeps this bound and the arithmetic safe. No collection moves inside the builder.

Normal preparation bypasses the stage entirely. Tests compare every normal
output byte and restore the two narrow profile hunks to compare the full
predecessor source. Shader, factory, selector and ABI changes are outside scope.
