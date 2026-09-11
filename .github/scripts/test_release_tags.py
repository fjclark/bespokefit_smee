# ruff: noqa: D100, D101, D102

import unittest
from pathlib import Path

from release_tags import classify, parse_tag


class ReleaseTagsTest(unittest.TestCase):
    def test_parses_canonical_release_tags(self):
        self.assertEqual(parse_tag("1.2.3"), ((1, 2, 3), True))
        self.assertEqual(parse_tag("1.2.3-rc.1"), ((1, 2, 3), False))

    def test_rejects_noncanonical_or_invalid_tags(self):
        for tag in ("latest", "v1.2.3", "1.2", "01.2.3", "1.2.3+build.1"):
            with self.subTest(tag=tag):
                self.assertIsNone(parse_tag(tag))

    def test_only_newest_stable_version_advances_latest(self):
        tags = ["0.8.1", "0.9.0-rc.1", "0.9.0", "not-a-version"]
        self.assertTrue(classify("0.9.0", tags)["is_latest"])
        self.assertFalse(classify("0.8.1", tags)["is_latest"])
        self.assertFalse(classify("0.9.0-rc.1", tags)["is_latest"])

    def test_only_newest_patch_advances_series_alias(self):
        tags = ["1.2.2", "1.2.3", "1.3.0"]
        self.assertTrue(classify("1.2.3", tags)["is_series_latest"])
        self.assertFalse(classify("1.2.2", tags)["is_series_latest"])

    def test_nonversion_tag_publishes_nothing(self):
        self.assertEqual(
            classify("nightly", ["1.0.0"]),
            {
                "is_version": False,
                "is_stable": False,
                "is_latest": False,
                "is_series_latest": False,
            },
        )

    def test_workflows_wire_release_guards(self):
        root = Path(__file__).parents[2]
        docker = (root / ".github/workflows/docker.yaml").read_text()
        docs = (root / ".github/workflows/docs.yaml").read_text()

        for workflow in (docker, docs):
            self.assertIn("fetch-depth: 0", workflow)
            self.assertIn("cancel-in-progress: false", workflow)
            self.assertIn(".github/scripts/release_tags.py", workflow)

        self.assertIn("flavor: latest=false", docker)
        self.assertIn("value=test-${{ inputs.tag }}", docker)
        self.assertIn("steps.release.outputs.is_version == 'true'", docker)
        self.assertIn("steps.release.outputs.is_latest == 'true'", docker)
        self.assertIn("steps.release.outputs.is_series_latest == 'true'", docker)
        self.assertIn("IS_LATEST: ${{ steps.release.outputs.is_latest }}", docs)


if __name__ == "__main__":
    unittest.main()
