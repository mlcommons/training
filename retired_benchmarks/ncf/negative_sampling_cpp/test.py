import torch
import negative_sampling

n_positives = 1000
n_users = 10
n_items = 500
users = torch.randint(size=[n_positives, 1], low=0, high=n_users)
items = torch.randint(size=[n_positives, 1], low=0, high=n_items)

positives = torch.cat([users, items], dim=1)
positives, _ = torch.sort(positives, dim=1)
positives, _ = torch.sort(positives, dim=0)

print("positives: ", positives)


sampler = negative_sampling.NegativeSampler(positives, n_users, n_items)
train_negatives = sampler.generate_train(4)
test_negatives = sampler.generate_test(20)

print(train_negatives)
print(test_negatives)

