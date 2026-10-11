"""Offline checks of the arms files: python3 -m unittest discover -s tests -v"""

import glob
import os
import re
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import survey  # noqa: E402

# Tool parsers registered in vLLM at 0b7f11a1e, the build in the pinned image (vllm/tool_parsers/__init__.py).
PARSERS = set(
    "dots deepseek_v3 deepseek_v31 deepseek_v32 deepseek_v4 deepseek_v41 cohere_command3 cohere_command4 ernie45 "
    "glm45 glm47 ling3 granite granite4 hermes poolside_v1 hunyuan_a13b hy_v3 hy_v4 internlm jamba lfm2 kimi_k2 "
    "kimi_k3 k2_horizon llama3_json llama4_json llama4_pythonic longcat mimo minimax_m2 minimax_m3 minicpm5 mistral "
    "olmo3 muse_glimmer openai phi4_mini_json pythonic qwen3_coder qwen3_xml seed_oss step3 step3p5 inkling xlam "
    "gigachat3 functiongemma gemma4 apertus".split()
)
MODES = {"load", "long", "gen"}
LINEUP = dict((m, s) for s, m in survey.LINEUP)
EXCLUDED = {"Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-8B"}


def arms(path):
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            model, tag, modes, *rest = line.split("=", 3)
            yield model, tag, modes, rest[0] if rest else ""


class ArmsFiles(unittest.TestCase):
    files = sorted(glob.glob(os.path.join(ROOT, "tpu", "arms-*.txt")))

    def test_files_exist(self):
        self.assertTrue(self.files)

    def test_every_arm_is_in_the_lineup_under_its_slot4(self):
        for f in self.files:
            for model, tag, _, _ in arms(f):
                with self.subTest(f=os.path.basename(f), model=model):
                    self.assertIn(model, LINEUP)
                    self.assertNotIn(model, EXCLUDED)
                    self.assertEqual(tag, LINEUP[model])
                    self.assertRegex(tag, r"^[a-z0-9]+_[a-z0-9]+$")

    def test_modes_and_flags(self):
        for f in self.files:
            for model, tag, modes, flags in arms(f):
                with self.subTest(tag=tag):
                    self.assertTrue(set(modes.split("+")) <= MODES, modes)
                    for flag in filter(None, flags.split(",")):
                        if flag.startswith("parser="):
                            self.assertIn(flag[len("parser="):], PARSERS)
                        elif flag.startswith("mm="):
                            self.assertTrue(set(flag[3:].split("+")) <= {"image", "audio", "video"}, flag)
                        elif flag.startswith("gmu="):
                            self.assertTrue(0 < float(flag[4:]) <= 1)
                        else:
                            self.assertEqual(flag, "nothink")

    def test_no_arm_twice_and_tiers_disjoint(self):
        seen = {}
        for f in self.files:
            for model, *_ in arms(f):
                self.assertNotIn(model, seen, f"{model} in {seen.get(model)} and {f}")
                seen[model] = f

    def test_qr_name_fits(self):
        # launch.sh names the queued resource ms-<yyyy-mm-dd>-<stem>-<hhmm>; the node id adds -node. 63 max.
        for f in self.files:
            stem = re.sub(r"^arms-|\.txt$", "", os.path.basename(f))
            self.assertLessEqual(len(f"ms-2026-10-11-{stem}-1200-node"), 63)


if __name__ == "__main__":
    unittest.main()
