from __future__ import annotations

import os
import tempfile
import unittest

from portfolio_policy_eval_v12_report import build_arg_parser, load_dotenv


class DotenvConfigTest(unittest.TestCase):
    def test_load_dotenv_overrides_process_env_and_skips_empty_values(self) -> None:
        old_value = os.environ.get("DOTENV_TEST_VALUE")
        old_empty = os.environ.get("DOTENV_EMPTY_VALUE")
        os.environ["DOTENV_TEST_VALUE"] = "from_process"
        os.environ["DOTENV_EMPTY_VALUE"] = "keep_me"
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as file:
                file.write("DOTENV_TEST_VALUE=from_file\n")
                file.write("DOTENV_EMPTY_VALUE=\n")
                file.write("export DOTENV_QUOTED_VALUE='quoted value'\n")
                path = file.name
            loaded = load_dotenv(path)
            self.assertEqual(loaded["DOTENV_TEST_VALUE"], "from_file")
            self.assertEqual(os.environ["DOTENV_TEST_VALUE"], "from_file")
            self.assertEqual(os.environ["DOTENV_EMPTY_VALUE"], "keep_me")
            self.assertEqual(os.environ["DOTENV_QUOTED_VALUE"], "quoted value")
        finally:
            if old_value is None:
                os.environ.pop("DOTENV_TEST_VALUE", None)
            else:
                os.environ["DOTENV_TEST_VALUE"] = old_value
            if old_empty is None:
                os.environ.pop("DOTENV_EMPTY_VALUE", None)
            else:
                os.environ["DOTENV_EMPTY_VALUE"] = old_empty
            os.environ.pop("DOTENV_QUOTED_VALUE", None)
            if "path" in locals():
                os.unlink(path)

    def test_engine_parser_defaults_can_come_from_environment(self) -> None:
        old_period = os.environ.get("ENGINE_YAHOO_PERIOD")
        old_port = os.environ.get("ENGINE_PORT")
        old_no_browser = os.environ.get("ENGINE_NO_BROWSER")
        os.environ["ENGINE_YAHOO_PERIOD"] = "5d"
        os.environ["ENGINE_PORT"] = "9999"
        os.environ["ENGINE_NO_BROWSER"] = "1"
        try:
            args = build_arg_parser().parse_args(["engine"])
            self.assertEqual(args.yahoo_period, "5d")
            self.assertEqual(args.port, 9999)
            self.assertTrue(args.no_browser)
        finally:
            for name, old in [
                ("ENGINE_YAHOO_PERIOD", old_period),
                ("ENGINE_PORT", old_port),
                ("ENGINE_NO_BROWSER", old_no_browser),
            ]:
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old


if __name__ == "__main__":
    unittest.main()
