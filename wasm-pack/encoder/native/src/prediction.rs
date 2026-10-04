//! Learn bounded, one-pass byte replacements from parser-assisted correspondences.
//! Replacements are guesses only. The delta always corrects the predicted dictionary.
use std::collections::{BTreeMap, HashMap};
use crate::{
    analysis::{self, Function},
    put_u32, Result, MAX_BYTES,
};
pub struct Prediction {
    pub dictionary: Vec<u8>,
    pub rules: Vec<u8>,
    pub rule_count: usize,
}

pub fn train(
    a: &[u8],
    fa: &[Function],
    b: &[u8],
    fb: &[Function],
) -> Result<Option<Prediction>> {
    let pairs = analysis::correspond(a, fa, b, fb);
    let mut votes: BTreeMap<Vec<u8>, BTreeMap<Vec<u8>, usize>> = BTreeMap::new();
    for (ai, bi, runs) in pairs {
        for (ap, bp, count) in runs {
            for (old, new) in fa[ai]
                .instructions[ap..ap + count]
                .iter()
                .zip(&fb[bi].instructions[bp..bp + count])
            {
                if old.relocatable {
                    *votes
                        .entry(a[old.start..old.end].to_vec())
                        .or_default()
                        .entry(b[new.start..new.end].to_vec())
                        .or_default() += 1;
                }
            }
        }
    }
    let mut rules = Vec::new();
    for (old, choices) in votes {
        let total: usize = choices.values().sum();
        // BTreeMap's byte ordering resolves equally supported alternatives.
        let Some((new, &support)) = choices
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))) else {
            continue;
        };
        if old != *new && support >= 2 && support * 10 >= total * 9
            && (3..=16).contains(&old.len()) && (1..=16).contains(&new.len())
        {
            rules.push((old, new.clone()));
        }
    }
    rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
    rules.truncate(65536);
    let mut counts = HashMap::new();
    rules
        .retain(|(old, _)| {
            let count = counts.entry(old[..3].to_vec()).or_insert(0);
            if *count == 256 {
                false
            } else {
                *count += 1;
                true
            }
        });
    if rules.is_empty() {
        return Ok(None);
    }
    let dictionary = apply(a, &rules)?;
    let before: usize = rules.iter().map(|r| r.0.len()).sum();
    let after: usize = rules.iter().map(|r| r.1.len()).sum();
    let mut encoded = b"PRD1".to_vec();
    for value in [a.len(), dictionary.len(), rules.len(), before, after] {
        put_u32(&mut encoded, value)?;
    }
    encoded.extend(rules.iter().map(|r| r.0.len() as u8));
    encoded.extend(rules.iter().map(|r| r.1.len() as u8));
    for (old, _) in &rules {
        encoded.extend(old);
    }
    for (_, new) in &rules {
        encoded.extend(new);
    }
    if encoded.len() > 4 * 1024 * 1024 {
        return Err("prediction rules exceed limit".into());
    }
    Ok(
        Some(Prediction {
            dictionary,
            rules: encoded,
            rule_count: rules.len(),
        }),
    )
}

fn apply(input: &[u8], rules: &[(Vec<u8>, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut buckets: HashMap<[u8; 3], Vec<usize>> = HashMap::new();
    for (i, (old, _)) in rules.iter().enumerate() {
        buckets.entry(old[..3].try_into()?).or_default().push(i);
    }
    let mut output = Vec::with_capacity(input.len());
    let mut at = 0;
    let mut work = 0usize;
    let budget = 64 * input.len() + 65536;
    while at < input.len() {
        let mut matched = None;
        if at + 3 <= input.len() {
            if let Some(candidates) = buckets
                .get(&<[u8; 3]>::try_from(&input[at..at + 3])?)
            {
                for &i in candidates {
                    let old = &rules[i].0;
                    if at + old.len() > input.len() {
                        continue;
                    }
                    let mut equal = true;
                    for j in 3..old.len() {
                        work += 1;
                        if work > budget {
                            return Err("prediction work budget exceeded".into());
                        }
                        if input[at + j] != old[j] {
                            equal = false;
                            break;
                        }
                    }
                    if equal {
                        matched = Some(i);
                        break;
                    }
                }
            }
        }
        let (bytes, consumed): (&[u8], usize) = match matched {
            Some(i) => (&rules[i].1, rules[i].0.len()),
            None => (&input[at..at + 1], 1),
        };
        if bytes.len() > MAX_BYTES - output.len() {
            return Err("predicted dictionary exceeds limit".into());
        }
        output.extend(bytes);
        at += consumed;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn longest_first_and_no_recursive_replacement() {
        let rules = vec![
            (b"abcd".to_vec(), b"abc".to_vec()), (b"abc".to_vec(), b"X".to_vec())
        ];
        assert_eq!(apply(b"abcdabc", & rules).unwrap(), b"abcX");
    }
    #[test]
    fn unsupported_prefix_is_unchanged() {
        assert_eq!(
            apply(b"xyzab", & [(b"abc".to_vec(), b"Y".to_vec())]).unwrap(), b"xyzab"
        );
    }
}
