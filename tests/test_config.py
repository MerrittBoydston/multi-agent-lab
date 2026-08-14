import tempfile
import unittest
from pathlib import Path

import yaml

from multi_agent_lab.config import WorkflowConfig, config_from_mapping, load_config


def valid_config():
    return {
        "version": 1,
        "stages": [
            {
                "id": "draft",
                "agents": [{
                    "id": "writer",
                    "role": "producer",
                    "provider": "mock",
                    "model": "mock-v1",
                    "system_prompt": "Draft",
                    "input": "{{ objective }}",
                }],
            },
            {
                "id": "decision",
                "agents": [{
                    "id": "judge",
                    "role": "reviewer",
                    "provider": "mock",
                    "model": "mock-v1",
                    "system_prompt": "Judge",
                    "input": "{{ artifact.draft.writer }}",
                }],
            },
        ],
    }


class ConfigTest(unittest.TestCase):
    def test_loads_and_round_trips_canonical_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workflow.yml"
            path.write_text(yaml.safe_dump(valid_config()), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.slots, (("draft", "writer"), ("decision", "judge")))
        self.assertEqual(WorkflowConfig.from_json(config.to_json()), config)

    def test_rejects_forward_and_unknown_template_references(self):
        forward = valid_config()
        forward["stages"][0]["agents"][0]["input"] = "{{ artifact.decision.judge }}"
        with self.assertRaisesRegex(ValueError, "unavailable artifact"):
            config_from_mapping(forward)

        unknown = valid_config()
        unknown["stages"][0]["agents"][0]["input"] = "{{ dangerous.code }}"
        with self.assertRaisesRegex(ValueError, "unsupported template"):
            config_from_mapping(unknown)

    def test_rejects_duplicate_agents_and_ambiguous_final_stage(self):
        duplicate = valid_config()
        duplicate["stages"][1]["agents"][0]["id"] = "writer"
        with self.assertRaisesRegex(ValueError, "duplicate agent"):
            config_from_mapping(duplicate)

        ambiguous = valid_config()
        ambiguous["stages"][1]["agents"].append({
            **ambiguous["stages"][1]["agents"][0], "id": "judge-b"
        })
        with self.assertRaisesRegex(ValueError, "final stage"):
            config_from_mapping(ambiguous)


if __name__ == "__main__":
    unittest.main()
