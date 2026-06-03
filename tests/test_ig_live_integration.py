"""
Opt-in IG platform integration tests.

These tests are skipped unless explicitly enabled with environment flags because they
connect to a real IG account. The live trade test can open a real Marvell position.

Required for platform smoke test:
    RUN_IG_PLATFORM_TEST=1
    IG_SERVICE_USERNAME
    IG_SERVICE_PASSWORD
    IG_SERVICE_API_KEY
    IG_SERVICE_ACC_TYPE
    IG_SERVICE_ACC_NUMBER
    IG_TEST_MARVELL_EPIC or IG_EPIC_MAP containing {"MRVL": "..."}

Required for the live Marvell buy test:
    RUN_IG_LIVE_TRADE_TEST=1
    IG_LIVE_TEST_SIZE=1   # optional; defaults to 1
"""

from __future__ import annotations

import os
import unittest

from portfolio_policy_eval_v12_report import IGClient, parse_ig_epic_map


if os.environ.get("RUN_IG_PLATFORM_TEST") != "1":
    class IGPlatformIntegrationDisabledTest(unittest.TestCase):
        @unittest.skip("Set RUN_IG_PLATFORM_TEST=1 to call the IG platform")
        def test_ig_platform_test_is_opt_in(self) -> None:
            pass
else:
    class IGPlatformIntegrationTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            cls.client = IGClient()
            if not cls.client.configured():
                raise unittest.SkipTest("IG service environment variables are incomplete")

            cls.currency_code = os.environ.get("IG_LIVE_TEST_CURRENCY", "USD")
            cls.expiry = os.environ.get("IG_LIVE_TEST_EXPIRY", "-")
            cls.client.login()

            cls.marvell_epic = os.environ.get("IG_TEST_MARVELL_EPIC") or parse_ig_epic_map("").get("MRVL")
            if not cls.marvell_epic:
                search = cls.client.search_markets(os.environ.get("IG_TEST_MARVELL_SEARCH", "Marvell"))
                markets = search.get("markets", [])
                cls.marvell_epic = next(
                    (item.get("epic", "") for item in markets if "marvell" in item.get("instrumentName", "").lower()),
                    "",
                )
            if not cls.marvell_epic:
                raise unittest.SkipTest("Set IG_TEST_MARVELL_EPIC or IG_EPIC_MAP with an MRVL entry")

        def test_can_login_and_read_marvell_market_snapshot(self) -> None:
            snapshot = self.client.market_snapshot(self.marvell_epic)
            self.assertTrue(snapshot)
            self.assertIn("snapshot", snapshot)

        def test_buy_one_marvell_position_when_explicitly_enabled(self) -> None:
            if os.environ.get("RUN_IG_LIVE_TRADE_TEST") != "1":
                raise unittest.SkipTest("Set RUN_IG_LIVE_TRADE_TEST=1 to place a live Marvell buy order")

            size = float(os.environ.get("IG_LIVE_TEST_SIZE", "1"))
            self.assertGreater(size, 0.0)

            ticket = self.client.create_market_position(
                epic=self.marvell_epic,
                direction="BUY",
                size=size,
                currency_code=self.currency_code,
                expiry=self.expiry,
                force_open=os.environ.get("IG_LIVE_TEST_FORCE_OPEN") == "1",
            )
            self.assertIn("dealReference", ticket)

            confirmation = self.client.confirm_deal(ticket["dealReference"])
            self.assertEqual(confirmation.get("dealStatus"), "ACCEPTED", confirmation)

            positions = self.client.open_positions()
            matching_positions = [
                item
                for item in positions.get("positions", [])
                if item.get("market", {}).get("epic") == self.marvell_epic
                and item.get("position", {}).get("direction") == "BUY"
            ]
            self.assertTrue(matching_positions, positions)


if __name__ == "__main__":
    unittest.main()
