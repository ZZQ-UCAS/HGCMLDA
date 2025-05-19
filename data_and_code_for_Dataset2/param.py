import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument('--data_path', type=str, default='./LDA_data2/',
                        help='the path of data.')

    parser.add_argument('--validation', type=int, default=5,
                        help='the fold number of CV.')

    parser.add_argument('--epoch', type=int, default=1600,
                        help='the number of epoch.')
                        
    parser.add_argument('--lnc_num', type=int, default=2853, 
                        help='the number of lncRNAs.')
    parser.add_argument('--dis_num', type=int, default=310,
                        help='the number of diseases.')

    parser.add_argument('--beta', type=int, default=0.04,
                        help='the size of beta.')

    return parser.parse_args()