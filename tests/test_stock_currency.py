"""Offline regression tests: currency and listed-share basis must not mix."""
import unittest
from unittest.mock import patch

from api.routers import stock_intelligence as stock


class CurrencyTests(unittest.TestCase):
    def setUp(self):
        self.info = dict(currency="USD", financialCurrency="USD", country="United States",
                         exchange="NMS", marketCap=1000, enterpriseValue=1100,
                         sharesOutstanding=10, currentPrice=100, revenueGrowth=.1,
                         forwardPE=20, priceToSalesTrailing12Months=2)
        self.model = {"records": [dict(revenue=500, net_income=50, free_cash_flow=40,
                                        net_margin=.1, net_cash=20)]}

    def test_us_issuer_keeps_derived_values(self):
        valuation = stock._valuation_snapshot(self.info, self.model)
        metrics = {m["key"]: m["value"] for m in valuation["metrics"]}
        self.assertEqual(metrics["fcf_yield"], .04)
        self.assertEqual(metrics["earnings_yield"], .05)
        self.assertEqual(metrics["price_sales"], 2)
        cases = stock._scenario_valuation(self.info, self.model)
        self.assertEqual(len(cases), 3)
        self.assertAlmostEqual(cases[1]["implied_price"], (500 * 1.1**5 * .1 * 25 + 20) / 10)

    def test_taiwan_reporting_currency_suppresses_mixed_values(self):
        self.info.update(financialCurrency="TWD", country="Taiwan", longName="Taiwan Semiconductor ADR")
        valuation = stock._valuation_snapshot(self.info, self.model)
        metrics = {m["key"]: m for m in valuation["metrics"]}
        for key in ("fcf_yield", "earnings_yield"):
            self.assertIsNone(metrics[key]["value"])
            self.assertIn("TWD", metrics[key]["unavailable_reason"])
        self.assertEqual(metrics["price_sales"]["value"], 2)
        self.assertEqual(metrics["price_sales"]["source"], "provider_reported")
        self.assertIsNone(valuation["reverse_expectations"]["required_revenue_cagr"])
        self.assertIsNone(valuation["reverse_expectations"]["required_year_5_net_income"])
        self.assertEqual(stock._scenario_valuation(self.info, self.model), [])

    def test_same_currency_adr_still_requires_share_reconciliation(self):
        self.info.update(country="United Kingdom", longName="Example ADS")
        validation = stock._valuation_currency_validation(self.info)
        self.assertEqual(validation["reason_codes"], ["share_basis_unverified"])
        self.assertEqual(stock._scenario_valuation(self.info, self.model), [])

    def test_missing_currency_does_not_default_to_usd(self):
        self.info.pop("financialCurrency")
        self.assertEqual(stock._scenario_valuation(self.info, self.model), [])
        profile = stock._company_profile({}, "TEST")
        self.assertIsNone(profile["currency"])
        self.assertIsNone(profile["financial_currency"])

    def test_minor_quote_unit_is_not_major_currency(self):
        self.info.update(currency="GBp", financialCurrency="GBP", country="United Kingdom", exchange="LSE")
        validation = stock._valuation_currency_validation(self.info)
        self.assertEqual(validation["quote_currency"], "GBp")
        self.assertIn("currency_mismatch", validation["reason_codes"])
        self.assertEqual(stock._scenario_valuation(self.info, self.model), [])

    def test_native_statement_units_and_eps_basis_are_labeled(self):
        model = {"rows": [{"key": "revenue", "format": "currency"},
                          {"key": "eps", "format": "number"}], "records": [{"revenue": 123}]}
        result = stock._label_financial_model(model, {"financialCurrency": "twd"})
        self.assertEqual(result["currency"], "TWD")
        self.assertEqual(result["rows"][1]["currency"], "TWD")
        self.assertEqual(result["records"][0]["revenue"], 123)

    def test_full_payload_wires_currency_metadata_without_network(self):
        import pandas as pd
        history = pd.DataFrame({"Close": [90, 100], "Open": [89, 99], "High": [91, 101],
                                "Low": [88, 98], "Volume": [1000, 1200]},
                               index=pd.date_range("2026-01-01", periods=2))
        self.info.update(financialCurrency="TWD", country="Taiwan")
        with patch.object(stock.yf, "Ticker"), patch.object(stock, "_safe_info", return_value=self.info), \
             patch.object(stock, "_safe_history", return_value=history), \
             patch.object(stock, "_build_financial_model", return_value=self.model), \
             patch.object(stock, "_fundamental_velocity", return_value={}), \
             patch.object(stock, "_revision_summary", return_value={}), \
             patch.object(stock, "_earnings_reactions", return_value={}), \
             patch.object(stock, "_options_snapshot", return_value={}), \
             patch.object(stock, "_build_peers", return_value=[]):
            payload = stock._full_stock_payload("TSM")
        self.assertEqual(payload["financial_model"]["currency"], "TWD")
        self.assertEqual(payload["scenario_validation"]["status"], "suppressed")
        self.assertEqual(payload["scenarios"], [])


if __name__ == "__main__":
    unittest.main()
