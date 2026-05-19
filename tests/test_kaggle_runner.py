from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_kaggle_pipeline import _find_prepared_data_dir, _install_prepared_data


class KaggleRunnerTests(unittest.TestCase):
    def test_auto_finds_prepared_data_folder_and_installs_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "input" / "ids2-preprocessed" / "data"
            (prepared / "interim").mkdir(parents=True)
            (prepared / "splits").mkdir()
            (prepared / "processed").mkdir()
            (prepared / "interim" / "cicids2017_clean.parquet").touch()
            (prepared / "interim" / "column_schema.json").write_text("{}", encoding="utf-8")

            found = _find_prepared_data_dir("auto", input_root=root / "input")
            self.assertEqual(found, prepared)

            project = root / "working" / "repo"
            project.mkdir(parents=True)
            _install_prepared_data(prepared, project, mode="symlink")

            self.assertTrue((project / "data" / "interim").is_symlink())
            self.assertTrue((project / "data" / "splits").is_symlink())
            self.assertTrue((project / "data" / "processed").is_symlink())
            self.assertTrue((project / "data" / "features").is_dir())


if __name__ == "__main__":
    unittest.main()
