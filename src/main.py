import argparse
from segDiniz import segDiniz

def main():
    #parser = argparse.ArgumentParser(description='segDiniz')
    #parser.add_argument('-c', '--conf', help='path to configuration file', required=True)

    #group = parser.add_mutually_exclusive_group()
    #group.add_argument('--train', action='store_true', help='Train')    
    #group.add_argument('--predict_on_test_set', action='store_true', help='Predict on test set')
    #group.add_argument('--predict', action='store_true', help='Predict on single file')

    #parser.add_argument('--filename', help='path to file')
    
    #args = parser.parse_args()

    seg = segDiniz()

    #if args.train:
    print('Starting training')
    seg.train()  
    #else:
    #    raise Exception('Unknown args') 

if __name__ == "__main__":
    main()

    