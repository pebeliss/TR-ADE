import argparse

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description='Default parameters of the models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches')
    parser.add_argument('--epochs',type=int, default=5000, help='Number of epochs')
    parser.add_argument('--latent_dim',type=int, default=3, help='Dimension of the latent space')
    parser.add_argument('--num_clusters',type=int, default=15, help='Number of clusters')
    parser.add_argument('--gamma',type=float, default=1, help='Weight of classification loss')
    parser.add_argument('--classify', type=bool, default=True, help='Classification.')
    parser.add_argument('--num_output_head',type=int, default=3, help='Number of output heads for classifier.')
    parser.add_argument('--num_classes',type=int, default=2, help='Number of classes per output head.')
    parser.add_argument('--validation_frac',type=float, default=0.2, help='Proportion of validation set')
    parser.add_argument('--imputation_strategy',type=str, default='MICE', help='Imputation strategy')
    parser.add_argument('--scaling_strategy',type=str, default='standard', help='Scaling strategy')
    parser.add_argument('--iqr_scaler',type=float, default=1.5, help='IQR scaler realted to outlier handling.')
    parser.add_argument('--data_path',type=str, default='../../data', help='Data folder')
    parser.add_argument('--results_path',type=str, default='../../results', help='Results folder')
    parser.add_argument('--use_early_stopping', type=bool, default=True, help='Usage of early stopping.')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Early stopping patience.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--use_lr_schedule', type=bool, default=True, help='Use decaying learning rate.')
    parser.add_argument('--s_to_classifier', type=bool, default=False, help='Give cluster assignment as input to classifier.')
    parser.add_argument('--learn_prior', type=bool, default=True, help='Optimize prior probabilities')
    parser.add_argument('--final_activation', type=str, default='softmax', help='Final activation for classifier.')

    return parser.parse_args(argv)