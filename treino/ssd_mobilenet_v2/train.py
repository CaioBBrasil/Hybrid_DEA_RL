from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tf_detection import train, train_parser

if __name__ == "__main__":
    train("ssd_mobilenet_v2", train_parser("ssd_mobilenet_v2").parse_args())

