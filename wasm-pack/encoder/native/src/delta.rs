//! WXP1 encoding. Long base copies may contain explicit additive byte corrections.
//! The browser applies corrections before later output-history copies read them.
use crate::{matcher::Record, put_u32, Result};
fn varint(out: &mut Vec<u8>, value: usize) -> Result<()> {
    let mut value = u32::try_from(value)?;
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
    Ok(())
}

fn merge(records: &[Record], base: &[u8], target: &[u8]) -> Vec<Record> {
    let mut output: Vec<Record> = Vec::new();
    let mut i = 0;
    while i < records.len() {
        let r = &records[i];
        let mut end = r.position + r.length;
        let mut last = i;
        if let Some(source) = r.source.filter(|&s| s + r.length <= base.len()) {
            let displacement = source as isize - r.position as isize;
            for (j, next) in records.iter().enumerate().skip(i + 1) {
                if next.position - end > 1024 {
                    break;
                }
                if let Some(s) = next.source {
                    if s + next.length <= base.len()
                        && s as isize - next.position as isize == displacement
                    {
                        let start = end as isize + displacement;
                        if start < 0 {
                            break;
                        }
                        let gap = next.position - end;
                        let mismatch = base[start as usize..start as usize + gap]
                            .iter()
                            .zip(&target[end..next.position])
                            .filter(|(a, b)| a != b)
                            .count();
                        if mismatch > 4usize.max(gap / 2) {
                            break;
                        }
                        end = next.position + next.length;
                        last = j;
                    }
                }
            }
        }
        let record = Record {
            position: r.position,
            length: end - r.position,
            source: r.source,
        };
        if let Some(previous) = output.last_mut() {
            let contiguous = match (previous.source, record.source) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a + previous.length == b && (a < base.len()) == (b < base.len())
                }
                _ => false,
            };
            if previous.position + previous.length == record.position && contiguous {
                previous.length += record.length;
                i = last + 1;
                continue;
            }
        }
        output.push(record);
        i = last + 1;
    }
    output
}

pub fn encode(base: &[u8], target: &[u8], records: &[Record]) -> Result<Vec<u8>> {
    let records = merge(records, base, target);
    let mut lanes = [Vec::new(), Vec::new(), Vec::new()];
    let mut literals: Vec<u8> = Vec::new();
    let mut gaps = Vec::new();
    let mut changes = Vec::new();
    let mut previous_change: isize = -1;
    let mut at = 0;
    let mut pending = 0;
    let mut count = 0;
    for r in records {
        if r.position != at || r.length > target.len() - at {
            return Err("invalid record coverage".into());
        }
        let Some(source) = r.source else {
            literals.extend(&target[at..at + r.length]);
            pending += r.length;
            at += r.length;
            continue;
        };
        if source < base.len() {
            if r.length > base.len() - source {
                return Err("copy crosses base boundary".into());
            }
            for i in 0..r.length {
                let correction = target[at + i].wrapping_sub(base[source + i]);
                if correction != 0 {
                    varint(&mut gaps, ((at + i) as isize - previous_change) as usize)?;
                    changes.push(correction);
                    previous_change = (at + i) as isize;
                }
            }
        } else {
            let earlier = source - base.len();
            if earlier >= at
                || target[earlier..earlier + r.length] != target[at..at + r.length]
            {
                return Err("nonexact output-history copy".into());
            }
        }
        varint(&mut lanes[0], pending)?;
        varint(&mut lanes[1], r.length)?;
        varint(&mut lanes[2], base.len() + at - source)?;
        count += 1;
        pending = 0;
        at += r.length;
    }
    if pending != 0 {
        varint(&mut lanes[0], pending)?;
        varint(&mut lanes[1], 0)?;
        varint(&mut lanes[2], 0)?;
        count += 1;
    }
    if at != target.len() {
        return Err("incomplete record coverage".into());
    }
    let mut output = b"WXP1".to_vec();
    for value in [
        4,
        base.len(),
        target.len(),
        count,
        lanes[0].len(),
        lanes[1].len(),
        lanes[2].len(),
        literals.len(),
        changes.len(),
        gaps.len(),
    ] {
        put_u32(&mut output, value)?;
    }
    for lane in lanes {
        output.extend(lane);
    }
    output.extend(literals);
    output.extend(gaps);
    output.extend(changes);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_varints() {
        let mut data = Vec::new();
        for n in [0, 127, 128, u32::MAX as usize] {
            varint(&mut data, n).unwrap();
        }
        assert_eq!(data, [0, 127, 128, 1, 255, 255, 255, 255, 15]);
        assert!(varint(& mut data, u32::MAX as usize + 1).is_err());
    }
    #[test]
    fn explicit_correction_keeps_changed_byte() {
        let base = b"aaaabbbbcccc";
        let target = b"aaaaxbbbcccc";
        let records = vec![
            Record { position : 0, length : 4, source : Some(0) }, Record { position : 4,
            length : 1, source : None }, Record { position : 5, length : 7, source :
            Some(5) }
        ];
        let encoded = encode(base, target, &records).unwrap();
        assert_eq!(u32::from_le_bytes(encoded[36..40].try_into().unwrap()), 1);
        assert_eq!(* encoded.last().unwrap(), b'x'.wrapping_sub(b'b'));
    }
}
