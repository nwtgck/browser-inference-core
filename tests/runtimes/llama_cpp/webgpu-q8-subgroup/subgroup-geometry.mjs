// Test-only observation. Production kernels use no readback or geometry probe.
export const subgroupGeometryCode = `enable subgroups;
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(subgroup_id) group: u32, @builtin(subgroup_invocation_id) lane: u32,
        @builtin(subgroup_size) width: u32, @builtin(num_subgroups) count: u32) {
    let base = local.x * 5u;
    output[base] = width; output[base+1u] = count;
    output[base+2u] = group; output[base+3u] = lane;
    output[base+4u] = subgroupAdd(1u);
}`;

export function checkSubgroupGeometry({ geometry, advertisedWidth }) {
  const require = (condition, message) => { if (!condition) throw new Error(message); };
  require(geometry.length === 128 * 5, 'Incomplete workgroup geometry');
  const groups = new Map(), reportedCount = geometry[1];
  require(reportedCount >= Math.ceil(128 / advertisedWidth) && reportedCount <= 128, 'Invalid subgroup count');
  for (let i = 0; i < 128; i++) {
    const [width, count, group, lane, active] = geometry.subarray(i * 5, i * 5 + 5);
    require(width === advertisedWidth, 'Builtin subgroup_size is outside the advertised fixed width');
    require(count === reportedCount && group < count, 'Inconsistent subgroup IDs/count');
    require(active >= 1 && active <= width && lane < active, 'Invalid active subgroup lane');
    if (!groups.has(group)) groups.set(group, { active, lanes: new Set() });
    const entry = groups.get(group);
    require(entry.active === active && !entry.lanes.has(lane), 'Inconsistent active count or duplicate lane');
    entry.lanes.add(lane);
  }
  require(groups.size === reportedCount, 'Incomplete subgroup ID partition');
  for (const { active, lanes } of groups.values()) {
    require(lanes.size === active && [...lanes].every(lane => lane < active), 'Incomplete dense subgroup lanes');
  }
  const activeCounts = [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([, entry]) => entry.active);
  require(activeCounts.reduce((sum, count) => sum + count, 0) === 128, 'Incomplete workgroup invocation partition');
  return { advertisedWidth, builtinWidth: advertisedWidth, numSubgroups: reportedCount, activeCounts,
    fullPacked: reportedCount === 128 / advertisedWidth, checkedInvocations: 128, ok: true };
}
