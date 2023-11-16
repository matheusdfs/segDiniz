import argparse
import logger
from segDiniz import segDiniz

def parseArguments():
    parser = argparse.ArgumentParser(description='segDiniz')

    parser.add_argument(
        '-d', 
        '--device', 
        help='configure device to train network', 
        required=True,
        choices=['cpu', 'cuda', 'mps'] ,
        default='cpu'
    )

    parser.add_argument(
        '-f', 
        '--checkpoint_file', 
        help='use last checkpoint', 
        required=False,
        choices=[True, False] ,
        default=True
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')    
    group.add_argument('--predict_on_test_set', action='store_true', help='Predict on test set')
    group.add_argument('--predict', action='store_true', help='Predict on single file')
    group.add_argument('--export', action='store_true', help='export ONNX file')
    
    return parser.parse_args()

def main():
    args = parseArguments()

    try:
        seg = segDiniz(args)
    except:
        logger.error("Error in the segmentation class initialization")

    if args.train:
        seg.train() 
    elif args.predict:
        seg.predict()
    else:
        raise Exception('Unknown args') 

if __name__ == "__main__":
    main()

    