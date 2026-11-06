from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tf_detection import test, test_parser

if __name__ == "__main__":
    test("ssd_mobilenet_v2", test_parser("ssd_mobilenet_v2").parse_args())

