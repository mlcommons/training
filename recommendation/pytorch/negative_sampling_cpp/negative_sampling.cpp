#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstdint>

struct NegativeSampler {
  NegativeSampler(at::Tensor positives, int n_user, int n_item) :
    positives_lists(n_user)
  {
    std::cout << "C++ PyTorch extension for negative sampling created." << std::endl;
    n_user_ = n_user;
    n_item_ = n_item;
    n_positives_ = positives.size(0);
    preprocessPositives(positives);
  }

  void preprocessPositives(at::Tensor positives) {
    auto positives_accessor = positives.accessor<int64_t, 2>();
    for (int i = 0; i != positives.size(0); ++i) {
      int user = positives_accessor[i][0];
      int item = positives_accessor[i][1];
      positives_lists[user].push_back(item);
    }
  }

  at::Tensor generate_test_part(int num_negatives, int chunk_size, int user_offset) {
    at::Tensor negatives = torch::empty({num_negatives * chunk_size, 2}, 
            torch::CPU(torch::kInt64));

    int i = 0;
    for (int u = 0; u != chunk_size; ++u) {
      int user = user_offset + u;
      for (int ni = 0; ni != num_negatives; ++ni) {
        bool is_positive = true;
        // repeat until a negative is found
        int item = -1;
        while (is_positive) {
          item = static_cast<int>(at::randint(0, n_item_ - 1, {1}, torch::kInt64).data<int64_t>()[0]);

          // check if the sampled number is a positive
          is_positive = std::binary_search(positives_lists[user].begin(), positives_lists[user].end(), item);
        }

        negatives[i][0] = user;
        negatives[i][1] = item;
        ++i;
      }
    }
    return negatives;
  }


  at::Tensor generate_train(int num_negatives) {
    at::Tensor negatives = torch::empty({num_negatives * n_positives_, 2}, torch::CPU(torch::kInt64));

    int i  = 0;
    for (int u = 0; u != n_user_; ++u) {
      // sample num_negatives for each positives for each user
      for (int ni = 0; ni != num_negatives; ++ni) {
        for (int pi = 0; pi != positives_lists[u].size(); ++pi) {
          bool is_positive = true;
          // repeat until a negative is found
          int item = -1;
          while (is_positive) {
            item = static_cast<int>(at::randint(0, n_item_ - 1, {1}, torch::kInt64).data<int64_t>()[0]);

            // check if the sampled number is a positive
            is_positive = std::binary_search(positives_lists[u].begin(),
                                             positives_lists[u].end(), item);
          }
    
          negatives[i][0] = u;
          negatives[i][1] = item;
          ++i;
        }
      }
    }
    return negatives;
  }

private:
  std::vector<std::vector<int>> positives_lists;
  int n_user_;
  int n_item_;
  int n_positives_;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<NegativeSampler>(m, "NegativeSampler")
    .def(py::init<at::Tensor, int, int>())
    .def("generate_train", &NegativeSampler::generate_train)
    .def("generate_test_part", &NegativeSampler::generate_test_part);
}
