//! Parser-assisted compression hints. No fingerprint proves semantic equivalence.
//! Original byte ranges are retained; no inference module is re-encoded.
use std::collections::BTreeMap;
use wasmparser::{Operator, Parser, Payload, Validator, WasmFeatures};
use crate::{Error, Result};
const MAX_FUNCTIONS: usize = 100_000;
const MAX_OPERATORS: usize = 16_000_000;
const MAX_FUNCTION_OPERATORS: usize = 4_000_000;
const SHINGLE: usize = 8;
#[derive(Clone, Debug)]
pub struct Instruction {
    pub start: usize,
    pub end: usize,
    pub shape: u64,
    pub strict: u64,
    pub relocatable: bool,
}

#[derive(Debug)]
pub struct Function {
    pub start: usize,
    pub end: usize,
    pub header_end: usize,
    pub instructions: Vec<Instruction>,
}

pub type Alignment = (usize, usize, usize);
pub type FunctionMatch = (usize, usize, Vec<Alignment>);
fn hash(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(
            0xcbf29ce484222325,
            |h, byte| { (h ^ u64::from(*byte)).wrapping_mul(0x100000001b3) },
        )
}

pub fn validate(bytes: &[u8]) -> Result<()> {
    if bytes.get(..8) != Some(b"\0asm\x01\0\0\0") {
        return Err("not a core WebAssembly module".into());
    }
    Validator::new_with_features(WasmFeatures::all()).validate_all(bytes)?;
    Ok(())
}

pub fn parse(bytes: &[u8]) -> Result<Vec<Function>> {
    validate(bytes)?;
    let mut functions = Vec::new();
    let mut total = 0;
    for payload in Parser::new(0).parse_all(bytes) {
        if let Payload::CodeSectionEntry(body) = payload? {
            let range = body.range();
            let mut reader = body.get_operators_reader()?;
            let header_end = usize::try_from(reader.original_position())?;
            let mut instructions = Vec::new();
            while !reader.eof() {
                if total >= MAX_OPERATORS || instructions.len() >= MAX_FUNCTION_OPERATORS
                {
                    return Err(Error::from("module operator budget exceeded"));
                }
                let start = usize::try_from(reader.original_position())?;
                let operator = reader.read()?;
                let end = usize::try_from(reader.original_position())?;
                let relocatable = matches!(
                    operator, Operator::Call { .. } | Operator::ReturnCall { .. } |
                    Operator::RefFunc { .. } | Operator::CallIndirect { .. } |
                    Operator::ReturnCallIndirect { .. } | Operator::GlobalGet { .. } |
                    Operator::GlobalSet { .. } | Operator::I32Const { .. } |
                    Operator::I64Const { .. }
                );
                let strict = hash(format!("{operator:?}").as_bytes());
                let shape = if relocatable {
                    hash(format!("{:?}", std::mem::discriminant(& operator)).as_bytes())
                } else {
                    strict
                };
                instructions
                    .push(Instruction {
                        start,
                        end,
                        shape,
                        strict,
                        relocatable,
                    });
                total += 1;
            }
            reader.finish()?;
            if functions.len() >= MAX_FUNCTIONS {
                return Err(Error::from("module function budget exceeded"));
            }
            functions
                .push(Function {
                    start: usize::try_from(range.start)?,
                    end: usize::try_from(range.end)?,
                    header_end,
                    instructions,
                });
        }
    }
    Ok(functions)
}

fn shape_key(function: &Function, bytes: &[u8]) -> (Vec<u8>, u64, usize) {
    let mut h: u64 = 0xcbf29ce484222325;
    for instruction in &function.instructions {
        for byte in instruction.shape.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    (bytes[function.start..function.header_end].to_vec(), h, function.instructions.len())
}

fn shingles(instructions: &[Instruction]) -> Vec<u64> {
    instructions
        .windows(SHINGLE)
        .map(|window| {
            window
                .iter()
                .enumerate()
                .fold(
                    0,
                    |h, (i, instruction)| {
                        let multiplier = 0x9e3779b185ebca87u64
                            .wrapping_add((i as u64).wrapping_mul(0x100000001b3));
                        h ^ instruction.shape.wrapping_mul(multiplier)
                    },
                )
        })
        .collect()
}

fn samples(values: &[u64]) -> Vec<u64> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.truncate(40);
    sorted
}

fn unique_positions(values: &[u64]) -> BTreeMap<u64, usize> {
    let mut positions = BTreeMap::new();
    for (i, &value) in values.iter().enumerate() {
        positions.entry(value).and_modify(|p| *p = usize::MAX).or_insert(i);
    }
    positions.retain(|_, position| *position != usize::MAX);
    positions
}

/// Align unique eight-instruction anchors, then extend without crossing neighbors.
fn align(
    a: &[Instruction],
    b: &[Instruction],
    ha: &[u64],
    hb: &[u64],
) -> Vec<Alignment> {
    let ua = unique_positions(ha);
    let ub = unique_positions(hb);
    let mut pairs: Vec<_> = ub
        .iter()
        .filter_map(|(h, &bi)| ua.get(h).map(|&ai| (ai, bi)))
        .collect();
    pairs.sort_unstable_by_key(|p| p.1);
    let mut tails = Vec::new();
    let mut indices = Vec::new();
    let mut previous = Vec::new();
    for (i, &(ai, _)) in pairs.iter().enumerate() {
        let at = tails.partition_point(|&tail| tail < ai);
        previous.push(if at == 0 { None } else { Some(indices[at - 1]) });
        if at == tails.len() {
            tails.push(ai);
            indices.push(i);
        } else {
            tails[at] = ai;
            indices[at] = i;
        }
    }
    let mut ordered = Vec::new();
    let mut current = indices.last().copied();
    while let Some(i) = current {
        ordered.push(pairs[i]);
        current = previous[i];
    }
    ordered.reverse();
    let mut runs: Vec<Alignment> = Vec::new();
    for (ai, bi) in ordered {
        if !a[ai..ai + SHINGLE]
            .iter()
            .zip(&b[bi..bi + SHINGLE])
            .all(|(a, b)| a.shape == b.shape)
        {
            continue;
            // Hashes only nominate candidates.
        }
        if let Some(last) = runs.last_mut() {
            if ai as isize - last.0 as isize == bi as isize - last.1 as isize
                && ai <= last.0 + last.2
            {
                last.2 = last.2.max(ai + SHINGLE - last.0);
                continue;
            }
            if ai < last.0 + last.2 || bi < last.1 + last.2 {
                continue;
            }
        }
        runs.push((ai, bi, SHINGLE));
    }
    let mut extended: Vec<Alignment> = Vec::new();
    for (i, &(mut ai, mut bi, mut count)) in runs.iter().enumerate() {
        let (left_a, left_b) = extended
            .last()
            .map_or((0, 0), |&(a, b, n)| (a + n, b + n));
        let (right_a, right_b) = runs
            .get(i + 1)
            .map_or((a.len(), b.len()), |&(a, b, _)| (a, b));
        while ai > left_a && bi > left_b && a[ai - 1].shape == b[bi - 1].shape {
            ai -= 1;
            bi -= 1;
            count += 1;
        }
        while ai + count < right_a && bi + count < right_b
            && a[ai + count].shape == b[bi + count].shape
        {
            count += 1;
        }
        if let Some(last) = extended.last_mut() {
            if last.0 + last.2 == ai && last.1 + last.2 == bi {
                last.2 += count;
                continue;
            }
        }
        extended.push((ai, bi, count));
    }
    extended
}

/// Full-shape matches first; partial functions use a bounded shortlist of anchors.
/// Deterministic tie-breaking is intentional; HashMap iteration must not affect output.
pub fn correspond(
    a: &[u8],
    fa: &[Function],
    b: &[u8],
    fb: &[Function],
) -> Vec<FunctionMatch> {
    let mut groups: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (i, f) in fa.iter().enumerate() {
        groups.entry(shape_key(f, a)).or_default().push(i);
    }
    let mut exact = BTreeMap::new();
    let mut result = Vec::new();
    for (j, f) in fb.iter().enumerate() {
        let Some(candidates) = groups.get(&shape_key(f, b)) else {
            continue;
        };
        let mut candidates = candidates.clone();
        if candidates.len() > 64 {
            candidates
                .sort_by_key(|&i| (
                    (fa[i].end - fa[i].start).abs_diff(f.end - f.start),
                    i.abs_diff(j),
                    i,
                ));
            candidates.truncate(64);
        }
        let winner = candidates
            .into_iter()
            .filter(|&i| {
                fa[i]
                    .instructions
                    .iter()
                    .zip(&f.instructions)
                    .all(|(a, b)| a.shape == b.shape)
            })
            .max_by_key(|&i| {
                let equal = fa[i]
                    .instructions
                    .iter()
                    .zip(&f.instructions)
                    .filter(|(a, b)| a.strict == b.strict)
                    .count();
                (
                    equal,
                    std::cmp::Reverse(
                        (fa[i].end - fa[i].start).abs_diff(f.end - f.start),
                    ),
                    std::cmp::Reverse(i.abs_diff(j)),
                    std::cmp::Reverse(i),
                )
            });
        if let Some(i) = winner {
            exact.insert(j, i);
            result.push((i, j, vec![(0, 0, f.instructions.len())]));
        }
    }
    let ha: Vec<_> = fa.iter().map(|f| shingles(&f.instructions)).collect();
    let mut inverted: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, hashes) in ha.iter().enumerate() {
        for h in samples(hashes) {
            inverted.entry(h).or_default().push(i);
        }
    }
    for (j, f) in fb.iter().enumerate() {
        if exact.contains_key(&j) {
            continue;
        }
        let hb = shingles(&f.instructions);
        let mut votes: BTreeMap<usize, f64> = BTreeMap::new();
        for h in samples(&hb) {
            if let Some(ids) = inverted.get(&h) {
                if ids.len() <= 80 {
                    for &i in ids {
                        *votes.entry(i).or_default() += 1.0 / ids.len() as f64;
                    }
                }
            }
        }
        for jj in j.saturating_sub(4)..(j + 5).min(fb.len()) {
            if let Some(&i) = exact.get(&jj) {
                let adjacent = i as isize + j as isize - jj as isize;
                if adjacent >= 0 && adjacent < fa.len() as isize {
                    *votes.entry(adjacent as usize).or_default() += 0.15;
                }
            }
        }
        let mut candidates: Vec<_> = votes.into_iter().collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.truncate(5);
        let mut best: Option<(usize, usize, f64, usize, Vec<Alignment>)> = None;
        for (i, vote) in candidates {
            let aa = &fa[i].instructions;
            let bb = &f.instructions;
            if aa.len().min(bb.len()) < SHINGLE
                || aa.len().max(bb.len()) > 4 * aa.len().min(bb.len())
            {
                continue;
            }
            let runs = align(aa, bb, &ha[i], &hb);
            let size: usize = runs
                .iter()
                .map(|&(_, bi, n)| bb[bi + n - 1].end - bb[bi].start)
                .sum();
            let replace = best
                .as_ref()
                .is_none_or(|(old_size, old_count, old_vote, old_i, _)| {
                    (size, std::cmp::Reverse(runs.len()))
                        > (*old_size, std::cmp::Reverse(*old_count))
                        || (size == *old_size && runs.len() == *old_count
                            && (vote.total_cmp(old_vote).is_gt()
                                || (vote == *old_vote
                                    && (i.abs_diff(j), i) < (old_i.abs_diff(j), *old_i))))
                });
            if replace {
                best = Some((size, runs.len(), vote, i, runs));
            }
        }
        if let Some((size, _, _, i, runs)) = best {
            if size >= 32 && size * 5 >= f.end - f.start {
                result.push((i, j, runs));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instructions(shapes: &[u64]) -> Vec<Instruction> {
        shapes.iter().enumerate().map(|(i, &shape)| Instruction {
            start: i, end: i + 1, shape, strict: shape, relocatable: false,
        }).collect()
    }

    #[test]
    fn partial_alignment_preserves_order_around_insertions() {
        let shapes: Vec<_> = (0..48).collect();
        let a = instructions(&shapes);
        let mut changed = shapes.clone();
        changed.splice(24..24, [999, 1000]);
        let b = instructions(&changed);
        let runs = align(&a, &b, &shingles(&a), &shingles(&b));
        assert_eq!(runs, [(0, 0, 24), (24, 26, 24)]);
        for &(ai, bi, count) in &runs {
            assert!(a[ai..ai + count].iter().zip(&b[bi..bi + count]).all(|(a, b)| a.shape == b.shape));
        }
    }

    #[test]
    fn hash_candidates_are_verified_as_instruction_shapes() {
        let a = instructions(&(0..16).collect::<Vec<_>>());
        let b = instructions(&(100..116).collect::<Vec<_>>());
        // Deliberately forged shingle hashes must not manufacture a match.
        let false_hashes: Vec<_> = (0..9).collect();
        assert!(align(&a, &b, &false_hashes, &false_hashes).is_empty());
    }

    #[test]
    fn validator_does_not_accept_an_arbitrary_binary_with_only_the_magic() {
        assert!(validate(b"\0asm\x01\0\0\0").is_ok());
        assert!(validate(b"\0asm\x01\0\0\0\x0a\xff").is_err());
    }
}
