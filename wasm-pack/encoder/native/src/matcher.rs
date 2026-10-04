//! Bounded exact matching over [base | target history]. Output copies may overlap.
//! There is no external compressor-specific sequence extraction dependency.
use crate::{Result, MAX_BYTES};
const HASH_BITS: usize = 21;
const NIL: u32 = u32::MAX;
const MAX_CHAIN: usize = 64;
#[derive(Clone, Debug)]
pub struct Record {
    pub position: usize,
    pub length: usize,
    /// Offset into the virtual [base | target] concatenation; None means literal.
    pub source: Option<usize>,
}

struct Matcher {
    history: Vec<u8>,
    base_len: usize,
    heads: Vec<u32>,
    previous: Vec<u32>,
    added: usize,
    repeated_offset: usize,
    work: usize,
    budget: usize,
}

impl Matcher {
    fn hash(&self, position: usize) -> usize {
        // Explicit byte order keeps output independent of the host architecture.
        let value = u32::from_le_bytes(
            self.history[position..position + 4].try_into().unwrap(),
        );
        (value.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize
    }
    fn insert(&mut self, until: usize) {
        let until = until.min(self.history.len().saturating_sub(3));
        while self.added < until {
            let key = self.hash(self.added);
            self.previous[self.added] = self.heads[key];
            self.heads[key] = self.added as u32;
            self.added += 1;
        }
    }
    fn length(&mut self, source: usize, position: usize) -> usize {
        let mut limit = self.history.len() - position;
        if source < self.base_len {
            limit = limit.min(self.base_len - source);
        }
        let mut n = 0;
        while n + 8 <= limit && self.work + 8 <= self.budget {
            self.work += 8;
            if self.history[source + n..source + n + 8]
                != self.history[position + n..position + n + 8]
            {
                break;
            }
            n += 8;
        }
        while n < limit && self.work < self.budget {
            self.work += 1;
            if self.history[source + n] != self.history[position + n] {
                break;
            }
            n += 1;
        }
        n
    }
    fn find(&mut self, position: usize) -> (usize, usize) {
        let mut best = (0, 0);
        if position + 4 > self.history.len() {
            return best;
        }
        if self.repeated_offset > 0 && self.repeated_offset <= position {
            let source = position - self.repeated_offset;
            let n = self.length(source, position);
            if n >= 4 {
                best = (n, source);
            }
        }
        let mut source = self.heads[self.hash(position)];
        for _ in 0..MAX_CHAIN {
            if source == NIL || self.work >= self.budget {
                break;
            }
            if (source as usize) < position {
                let source = source as usize;
                let n = self.length(source, position);
                if n > best.0 || (n == best.0 && source > best.1) {
                    best = (n, source);
                }
                if n == self.history.len() - position {
                    break;
                }
            }
            source = self.previous[source as usize];
        }
        if best.0 < 4 {
            best.0 = 0;
        }
        best
    }
}

pub fn find_matches(base: &[u8], target: &[u8]) -> Result<Vec<Record>> {
    if base.len() > MAX_BYTES || target.len() > MAX_BYTES {
        return Err("matcher input limit".into());
    }
    let mut history = base.to_vec();
    history.extend(target);
    let mut matcher = Matcher {
        previous: vec![NIL; history.len()],
        heads: vec![NIL; 1 << HASH_BITS],
        base_len: base.len(),
        budget: 256 * history.len() + 65536,
        history,
        added: 0,
        repeated_offset: 0,
        work: 0,
    };
    matcher.insert(base.len());
    let mut position = 0;
    let mut literal_start = 0;
    let mut records = Vec::new();
    while position < target.len() {
        matcher.insert(base.len() + position);
        let (mut n, mut source) = matcher.find(base.len() + position);
        if n >= 4 && position + 1 < target.len() {
            matcher.insert(base.len() + position + 1);
            if matcher.find(base.len() + position + 1).0 > n + 1 {
                position += 1;
                continue;
            }
        }
        if n < 4 {
            position += 1;
            continue;
        }
        while position > literal_start && source > 0 && source != base.len()
            && matcher.history[source - 1] == target[position - 1]
        {
            source -= 1;
            position -= 1;
            n += 1;
        }
        if position > literal_start {
            records
                .push(Record {
                    position: literal_start,
                    length: position - literal_start,
                    source: None,
                });
        }
        records
            .push(Record {
                position,
                length: n,
                source: Some(source),
            });
        matcher.repeated_offset = base.len() + position - source;
        position += n;
        literal_start = position;
    }
    if literal_start < target.len() {
        records
            .push(Record {
                position: literal_start,
                length: target.len() - literal_start,
                source: None,
            });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arbitrary_bytes_and_overlapping_copies() {
        for n in [0, 1, 3, 4, 17, 256, 4096] {
            let base: Vec<_> = (0..n).map(|i| (i % 31) as u8).collect();
            let mut target = base.clone();
            target.extend(b"aaaaaaaaaaaaaaaaaaaa");
            target.extend(&base);
            let records = find_matches(&base, &target).unwrap();
            let mut out = Vec::new();
            for r in records {
                assert_eq!(r.position, out.len());
                match r.source {
                    None => out.extend(&target[r.position..r.position + r.length]),
                    Some(s) => {
                        for i in 0..r.length {
                            out.push(
                                if s + i < base.len() {
                                    base[s + i]
                                } else {
                                    out[s + i - base.len()]
                                },
                            );
                        }
                    }
                }
            }
            assert_eq!(out, target);
        }
    }
}
