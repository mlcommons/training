import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

from load import implicit_load


USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'

NUMBER_NEGATIVES = 100

TRAIN_RATINGS_FILENAME = 'train-ratings.csv'
TEST_RATINGS_FILENAME = 'test-ratings.csv'
TEST_NEG_FILENAME = 'test-negative.csv'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def main():
    # TODO: Add random seed as parameter
    np.random.seed(0)
    args = parse_args()

    df = implicit_load(args.path, sort=False)
    grouped = df.groupby(USER_COLUMN)
    df = grouped.filter(lambda x: len(x) >= 20)

    original_users = df[USER_COLUMN].unique()
    original_items = df[ITEM_COLUMN].unique()

    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
    df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])

    assert df[USER_COLUMN].max() == len(original_users) - 1
    assert df[ITEM_COLUMN].max() == len(original_items) - 1

    # Need to sort before popping to get last item
    df.sort_values(by='timestamp', inplace=True)
    all_ratings = set(zip(df[USER_COLUMN], df[ITEM_COLUMN]))
    user_to_items = defaultdict(list)
    for row in df.itertuples():
        user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))  # noqa: E501

    test_ratings = []
    test_negs = []
    all_items = set(range(len(original_items)))
    for user in range(len(original_users)):
        test_item = user_to_items[user].pop()

        all_ratings.remove((user, test_item))
        all_negs = all_items - set(user_to_items[user])
        all_negs = sorted(list(all_negs))  # determinism

        test_ratings.append((user, test_item))
        test_negs.append(list(np.random.choice(all_negs, NUMBER_NEGATIVES)))

    # serialize
    df_train_ratings = pd.DataFrame(list(all_ratings))
    df_train_ratings['fake_rating'] = 1
    df_train_ratings.to_csv(os.path.join(args.output, TRAIN_RATINGS_FILENAME),
                            index=False, header=False, sep='\t')

    df_test_ratings = pd.DataFrame(test_ratings)
    df_test_ratings['fake_rating'] = 1
    df_test_ratings.to_csv(os.path.join(args.output, TEST_RATINGS_FILENAME),
                           index=False, header=False, sep='\t')

    df_test_negs = pd.DataFrame(test_negs)
    df_test_negs.to_csv(os.path.join(args.output, TEST_NEG_FILENAME),
                        index=False, header=False, sep='\t')


if __name__ == '__main__':
    main()
