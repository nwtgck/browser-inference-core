"""Shared network-free identities reject ambiguous and endpoint-changing values."""
import unittest
from build_tools.github import full_sha, repository_name, positive_id


class GitHubIdentity(unittest.TestCase):
    def test_repository_names_are_two_literal_components(self):
        self.assertEqual(repository_name('example/core-name'), 'example/core-name')
        for value in ('owner/repo?x=y', 'owner/repo/extra', 'owner\n/repo', '../repo', 'owner/..',
                      './repo', 'https://elsewhere', None, [], 123):
            with self.subTest(value=value), self.assertRaises(ValueError): repository_name(value)

    def test_commit_identities_require_full_lowercase_hex(self):
        self.assertEqual(full_sha('a' * 40), 'a' * 40)
        for value in ('abc', 'main', 'A' * 40, 'a' * 40 + '?page=2', None, 123, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError): full_sha(value)

    def test_ids_are_strict_positive_integers(self):
        self.assertEqual(positive_id(7), 7)
        for value in (0, -1, True, 1.0, 1.5, '7/comments', '42', None):
            with self.subTest(value=value), self.assertRaises(ValueError): positive_id(value)


if __name__ == '__main__': unittest.main()
