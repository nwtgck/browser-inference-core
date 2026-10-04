//! Bounded streaming bridge to unmodified ruzstd.
//!
//! State belongs to one decoder instance. JavaScript asks Rust to allocate its
//! input buffers, fills the returned ranges, then drains output in fixed chunks.
//! No pointer supplied by JavaScript is dereferenced in this implementation.
use std::{cell::RefCell, io::Read};
use ruzstd::decoding::{BlockDecodingStrategy, Dictionary, FrameDecoder};
const LIMIT: usize = 64 * 1024 * 1024;
const CHUNK: usize = 64 * 1024;
#[derive(Default)]
struct State {
    input: Vec<u8>,
    base: Vec<u8>,
    chunk: Vec<u8>,
    frame: Option<FrameDecoder>,
    consumed: usize,
    produced: usize,
    expected: usize,
    done: bool,
}
thread_local! {
    static STATE : RefCell < State > = RefCell::new(State::default());
}
/// Discard a successful or failed operation before preparing another one.
#[unsafe(no_mangle)]
pub extern "C" fn clear() {
    STATE.with(|state| *state.borrow_mut() = State::default());
}
/// Return a decoder-owned buffer. kind=0 is compressed input; kind=1 is a prefix.
/// Zero means invalid size/state. The caller must not retain views across growth.
#[unsafe(no_mangle)]
pub extern "C" fn prepare(kind: u32, size: usize) -> usize {
    if size > LIMIT || kind > 1 {
        return 0;
    }
    STATE
        .with(|state| {
            let mut state = state.borrow_mut();
            if state.frame.is_some() {
                return 0;
            }
            let buffer = if kind == 0 { &mut state.input } else { &mut state.base };
            buffer.resize(size, 0);
            buffer.as_mut_ptr() as usize
        })
}
/// Start one frame. Nonzero status is failure; the caller then clears the state.
#[unsafe(no_mangle)]
pub extern "C" fn start(expected: usize, max_window: usize) -> u32 {
    if expected > LIMIT || max_window > LIMIT || max_window == 0 {
        return 1;
    }
    STATE
        .with(|state| {
            let mut state = state.borrow_mut();
            if state.input.is_empty() || state.frame.is_some() {
                return 2;
            }
            let mut frame = FrameDecoder::new();
            frame.set_max_window_size(max_window as u64);
            let mut input = &state.input[..];
            if frame.init(&mut input).is_err() {
                return 3;
            }
            let consumed = state.input.len() - input.len();
            if frame.content_size() != expected as u64 {
                return 4;
            }
            if !state.base.is_empty() {
                let dictionary = Dictionary {
                    id: 1,
                    fse: Default::default(),
                    huf: Default::default(),
                    dict_content: std::mem::take(&mut state.base),
                    offset_hist: [1, 4, 8],
                };
                if frame.add_dict(dictionary).is_err() || frame.force_dict(1).is_err() {
                    return 5;
                }
            }
            state.expected = expected;
            state.produced = 0;
            state.consumed = consumed;
            state.done = false;
            state.chunk.resize(CHUNK, 0);
            state.frame = Some(frame);
            0
        })
}
/// Output remains owned by Rust; copy it before the next pull or clear.
#[unsafe(no_mangle)]
pub extern "C" fn chunk_ptr() -> usize {
    STATE.with(|state| state.borrow().chunk.as_ptr() as usize)
}
/// Positive is output length; zero is verified end-of-frame; negative is failure.
#[unsafe(no_mangle)]
pub extern "C" fn pull() -> i32 {
    STATE
        .with(|state| {
            let mut state = state.borrow_mut();
            if state.done {
                return 0;
            }
            let State { input, chunk, frame, consumed, produced, expected, done, .. } = &mut *state;
            let Some(frame) = frame.as_mut() else {
                return -1;
            };
            let mut iterations = 0;
            while frame.can_collect() == 0 && !frame.is_finished() {
                if iterations > input.len() / 3 + 1 {
                    return -2;
                }
                iterations += 1;
                let mut bytes = &input[*consumed..];
                let before = bytes.len();
                if frame
                    .decode_blocks(&mut bytes, BlockDecodingStrategy::UptoBlocks(1))
                    .is_err()
                {
                    return -3;
                }
                let used = before - bytes.len();
                if used == 0 {
                    return -4;
                }
                *consumed += used;
            }
            let n = match frame.read(chunk) {
                Ok(n) => n,
                Err(_) => return -5,
            };
            if n > *expected - *produced {
                return -6;
            }
            *produced += n;
            if frame.is_finished() && frame.can_collect() == 0 {
                if *produced != *expected || *consumed != input.len() {
                    return -7;
                }
                if let Some(checksum) = frame.get_checksum_from_data() {
                    if frame.get_calculated_checksum() != Some(checksum) {
                        return -8;
                    }
                }
                *done = true;
            }
            if n == 0 && !*done {
                return -9;
            }
            n as i32
        })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_state_and_limits() {
        clear();
        assert_eq!(pull(), - 1);
        assert_eq!(prepare(2, 1), 0);
        assert_eq!(prepare(0, LIMIT + 1), 0);
        assert_eq!(start(0, 0), 1);
        assert_eq!(start(0, LIMIT), 2);
        assert_ne!(prepare(0, 4), 0);
        assert_eq!(start(4, LIMIT), 3);
        clear();
        assert_eq!(pull(), - 1);
    }
}
