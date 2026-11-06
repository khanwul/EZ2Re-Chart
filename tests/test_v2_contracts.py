from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from ez2rc.v2.config import ConfigError, load_song
from ez2rc.v2.contracts import Chart, Event, Metadata, Note, Timing, TimeInterval, read_ratio
from ez2rc.v2.export import chart_payload, read_chart, read_events, write_outputs
from ez2rc.v2.storage import write_json
from ez2rc.v2.timing import beat_to_ms, ms_to_beat


class V2ContractsTest(unittest.TestCase):
    def test_decimal_bpm_and_negative_origin_are_exact(self):
        timing = Timing("191.910", -125.5, Fraction(4), "resolved", None)
        for beat in (Fraction(1, 7), Fraction(7, 11), Fraction(987, 5)):
            self.assertEqual(ms_to_beat(beat_to_ms(beat, timing), timing), beat)
        self.assertEqual(chart_payload(Chart(Metadata("g", "5k", 5, "t", "SHD"), timing))
                         ["timing"]["bpm_timeline"][0]["bpm"], "191.910")

    def test_complete_cannot_hide_unknown_origin_or_unresolved_event(self):
        metadata = Metadata("g", "5k", 5, "title", "SHD")
        event = Event("e1", 0, "incomplete", reason="tail clipped", candidate_ids=("c1",))
        chart = Chart(metadata, Timing("200"), unresolved=(event,))
        self.assertEqual(chart.status, "partial")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_outputs(root, chart, (event,), observation_ref="cache")
            payload = read_chart(root / "chart.json")
            self.assertIsNone(payload["timing"]["beat_zero_ms"])
            self.assertEqual(payload["unresolved"][0]["candidate_refs"], ["c1"])
            observed = read_events(root / "artifacts/events.json")
            self.assertEqual(observed["events"][0]["candidate_ids"], ["c1"])
            payload["status"] = "complete"
            write_json(root / "chart.json", payload)
            with self.assertRaisesRegex(ValueError, "completion"):
                read_chart(root / "chart.json")

    def test_rational_and_lane_and_note_invariants(self):
        for pair in ([2, 4], [1, 0], [1, -2], [True, 2], [1.0, 2]):
            with self.assertRaises(ValueError):
                read_ratio(pair)
        with self.assertRaises(ValueError):
            Note("n1", 0, "longnote", Fraction(3), Fraction(3))
        with self.assertRaises(ValueError):
            Note("n1", 0, "tap", Fraction(3), Fraction(4))
        with self.assertRaises(ValueError):
            Chart(Metadata("g", "5k", 5, "t", "SHD"), Timing("200"),
                  (Note("n1", 5, "tap", Fraction(0)),))
        with self.assertRaises(ValueError):
            Chart(Metadata("g", "5k", 5, "t", "SHD"), Timing("200"),
                  (Note("n1", 0, "tap", Fraction(0)),),
                  (Event("n1", 0, "ambiguous", reason="type"),))

    def test_endpoint_uncertainties_are_independent(self):
        event = Event("n", 0, "resolved", "longnote", 1, 40, TimeInterval(0, 2), TimeInterval(35, 45))
        self.assertNotEqual(event.start_interval, event.end_interval)
        with self.assertRaises(ValueError):
            Event("n", 0, "resolved", "longnote", 1, 40, TimeInterval(2, 4))

    def test_v1_input_and_future_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.toml").write_text('[setup]\nskin="x"\n')
            with self.assertRaisesRegex(ConfigError, "schema_version"):
                load_song(root / "old.toml")
            (root / "chart.json").write_text(json.dumps({"format": "ez2rc.chart", "schema_version": 99}))
            with self.assertRaisesRegex(ValueError, "schema"):
                read_chart(root / "chart.json")


if __name__ == "__main__":
    unittest.main()
