#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <unordered_set>

namespace py = pybind11;

std::tuple<std::vector<int>, std::vector<int>, std::vector<std::vector<int>>>
sample_batch(
    const std::vector<std::pair<int, std::vector<int>>> &batch,
    int num_users,
    int num_negatives
) {
    std::vector<int> item_list;
    std::vector<int> pos_user_list;
    std::vector<std::vector<int>> neg_user_list;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> user_dist(0, num_users - 1);

    for (const auto &[item, sampled_users_vec] : batch) {
        item_list.push_back(item);

        // 正样本随机选一个
        std::uniform_int_distribution<> pos_dist(0, sampled_users_vec.size() - 1);
        int pos_user = sampled_users_vec[pos_dist(gen)];
        pos_user_list.push_back(pos_user);

        // 负样本采样
        std::unordered_set<int> sampled_users_set(sampled_users_vec.begin(), sampled_users_vec.end());
        std::vector<int> neg_users;
        while (neg_users.size() < (size_t)num_negatives) {
            int neg_user = user_dist(gen);
            if (sampled_users_set.find(neg_user) == sampled_users_set.end()) {
                neg_users.push_back(neg_user);
            }
        }
        neg_user_list.push_back(neg_users);
    }

    return std::make_tuple(item_list, pos_user_list, neg_user_list);
}

PYBIND11_MODULE(cpp_sampler, m) {
    m.def("sample_batch", &sample_batch, "Sample positive and negative users",
        py::arg("batch"), py::arg("num_users"), py::arg("num_negatives"));
}
