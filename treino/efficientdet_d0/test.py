from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tf_detection import test, test_parser

if __name__ == "__main__":
    test("efficientdet_d0", test_parser("efficientdet_d0").parse_args())

