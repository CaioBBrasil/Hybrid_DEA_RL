from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tf_detection import train, train_parser

if __name__ == "__main__":
    train("efficientdet_d0", train_parser("efficientdet_d0").parse_args())

