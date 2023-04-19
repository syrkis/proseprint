# main.py
#   mawsa — multi author writing style analysis
# by: Noah Syrkis

# imports
from src import get_data, get_args, run_baseline, run_gru_siam


# main
def main():
    args = get_args()
    train_dataset = get_data(args.dataset, split='train')
    valid_dataset = get_data(args.dataset, split='validation')

    if args.model == 'baseline':
        run_baseline(train_dataset, valid_dataset)

    if args.model =='gru' and args.process == 'siam':
        conf = {'embedding_dim': 100, 'hidden_dim': 100, 'vocab_size': 100, 'tagset_size': 100, 'lr': 0.01}
        run_gru_siam(train_dataset, valid_dataset, conf)

    # if args.model == 'gru' and args.process == 'hirac':
    #     run_gru_hirac(train_dataset, valid_dataset)


if __name__ == '__main__':
    main()
    