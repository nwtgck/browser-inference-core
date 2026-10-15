"""Independent mask-level checks; not a substitute for a native/browser test.

The original path's liveness rule is clamp(score + 1, 0, 1). In this model,
visible scores are nonnegative and invisible scores are negative infinity.
The ordering below intentionally differs between ranked and unranked paths.
"""
import itertools
import math
import random
import struct
import unittest


def selection(scores, pool_cells, tail_cells, kq_mask, budget):
    selected = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:budget]
    live_cells = set(tail_cells)
    for index in selected:
        live = min(1.0, max(0.0, scores[index] + 1.0))
        # Invalid entries are redirected to discarded dump slots by the native
        # graph; their real cells must never become visible.
        if live == 1.0:
            live_cells.update(pool_cells[index])
        elif live != 0.0:
            raise ValueError('The Qwen4Exp score invariant was violated')
    return tuple(value if cell in live_cells else -math.inf
                 for cell, value in enumerate(kq_mask))


class AllPoolsEquivalence(unittest.TestCase):
    def test_every_visibility_pattern_and_tail(self):
        rng = random.Random(81371)
        for count in range(1, 9):
            for visible in itertools.product((False, True), repeat=count):
                for pooling in (1, 2, 4):
                    cells = [list(range(i * pooling, (i + 1) * pooling)) for i in range(count)]
                    rng.shuffle(cells)  # storage-cell order need not equal pool order
                    scores = [rng.choice((0.0, 1e-8, 0.5, 5.0, 1e20)) if live else -math.inf for live in visible]
                    unranked = [0.0 if live else -math.inf for live in visible]
                    # Holes, additive finite biases and another sequence's cells.
                    mask = [rng.choice((0.0, -0.25, -math.inf)) for _ in range(count * pooling + pooling)]
                    for tail in ([], [len(mask) - 1], list(range(len(mask) - pooling, len(mask)))):
                        self.assertEqual(selection(scores, cells, tail, mask, count),
                                         selection(unranked, cells, tail, mask, count))

    def test_it_is_not_a_dense_attention_substitution(self):
        mask = [0.0] * 10
        actual = selection([2.0, -math.inf], [[1, 3], [4, 6]], [9], mask, 2)
        self.assertEqual(actual, (-math.inf, 0.0, -math.inf, 0.0, -math.inf,
                                  -math.inf, -math.inf, -math.inf, -math.inf, 0.0))
        self.assertNotEqual(actual, tuple(mask))

    def test_ranking_is_required_above_the_selection_budget(self):
        cells = [[0], [1], [2]]
        original = selection([0.0, 1.0, 5.0], cells, [], [0.0] * 3, 2)
        unsafe = selection([0.0, 0.0, 0.0], cells, [], [0.0] * 3, 2)
        self.assertNotEqual(original, unsafe)

    def test_negative_scores_are_not_a_generic_indexer_optimization(self):
        with self.assertRaises(ValueError):
            selection([-0.5], [[0]], [], [0.0], 1)

    def test_reordering_needs_exact_float32_dump_rows(self):
        def f32(value):
            return struct.unpack('<f', struct.pack('<f', value))[0]

        def redirect(cell, n_kv, slot):
            # Match the graph's casts, cumsum + bias, subtraction and addition.
            dump = f32(f32(slot + 1) + f32(n_kv - 1))
            return int(f32(f32(f32(cell) - dump) + dump))

        # Every integer in the reviewed range is exactly representable. Check
        # several boundaries without constructing a multi-gigabyte model.
        for n_kv in (256, (1 << 23), (1 << 24) - 8):
            for cell in (0, 1, 3, n_kv // 2, n_kv - 1):
                self.assertEqual([redirect(cell, n_kv, slot) for slot in range(8)], [cell] * 8)
        # Without the shape guard, merely changing a member's slot can change
        # the address after F32 cancellation, even though the member is live.
        self.assertNotEqual(redirect(1, 1 << 24, 0), redirect(1, 1 << 24, 2))

    def test_shared_selection_must_keep_mask_identity(self):
        first = selection([2.0], [[0, 1]], [], [0.0, -math.inf], 1)
        other = selection([2.0], [[0, 1]], [], [-math.inf, 0.0], 1)
        self.assertNotEqual(first, other)


if __name__ == '__main__':
    unittest.main(verbosity=2)
