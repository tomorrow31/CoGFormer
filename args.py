import argparse
def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help="Device: cuda:(num) or (cpu)")
    parser.add_argument("--path", type=str, default='/lzy/CoGFormer/data/', help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="ALOI", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=2, help="Number of repeated times. Default is 5.")


    parser.add_argument("--knns", type=int, default=15, help="Number of k nearest neighbors")
    parser.add_argument("--common_neighbors", type=int, default=2, help="Number of common neighbors (when using pruning strategy 2)")
    parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")
    parser.add_argument("--ghost", action='store_true', default=False, help="xx")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")#1e-3
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--share_lr", type=float, default=1e-3, help="share_z_Learning rate")
    parser.add_argument("--share_weight_decay", type=float, default=5e-5, help="share_z_Weight decay")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of training epochs. Default is 1000.")
    parser.add_argument("--hdim", type=int, default=256, help="Number of hidden dimensions")


    parser.add_argument('--num_centroids', type=int, default=128, help="Number of centroids")



    args = parser.parse_args()

    return args
