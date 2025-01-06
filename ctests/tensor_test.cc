#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "tensor.h"

// ========= TESTs =========
TEST(TensorTest, IntializeAndCheckGetSet) {
  int n_of_rows = 5;
  int n_of_cols = 3;

  std::unique_ptr<Tensor> t =
      std::make_unique<Tensor>(std::vector<int>{n_of_rows, n_of_cols});
  // set tensor
  for (int i = 0; i < n_of_rows; i++) {
    for (int j = 0; j < n_of_cols; j++) {
      std::shared_ptr<Value> _v =
          std::make_shared<Value>(i * n_of_rows + j * n_of_cols);
      t->set(std::vector<int>{i, j}, _v);
    }
  }

  // get tensor
  for (int i = 0; i < n_of_rows; i++) {
    for (int j = 0; j < n_of_cols; j++) {
      double expected_data = (i * n_of_rows + j * n_of_cols);
      double got_data = t->get({i, j})->data;

      EXPECT_DOUBLE_EQ(got_data, expected_data);
    }
  }

  //   check if dimension and shape is correct
  EXPECT_EQ(t->dims(), 2);

  std::vector<int> expected_shape = {5, 3};
  std::vector<int> got_shape = t->shape;
  for (int i = 0; i < int(got_shape.size()); i++) {
    EXPECT_EQ(got_shape[i], expected_shape[i]);
  }
}

TEST(TensorTest, TestNormalizeIdx) {
  // ======== 1-d testing ========
  std::unique_ptr<Tensor> t1 = std::make_unique<Tensor>(std::vector<int>{5});

  for (int i = 0; i < 5; i++) {
    int norm_idx = t1->normalize_idx(std::vector<int>{i});
    EXPECT_EQ(norm_idx, i);
  }

  // ======== 2-d testing ========
  std::unique_ptr<Tensor> t2 = std::make_unique<Tensor>(std::vector<int>{5, 3});

  int my_real_idx = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      int norm_idx = t2->normalize_idx(std::vector<int>{i, j});
      EXPECT_EQ(norm_idx, my_real_idx);
      my_real_idx++;
    }
  }

  // ======== 3-d testing ========
  std::unique_ptr<Tensor> t3 =
      std::make_unique<Tensor>(std::vector<int>{5, 3, 2});

  my_real_idx = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 2; k++) {
        int norm_idx = t3->normalize_idx(std::vector<int>{i, j, k});
        EXPECT_EQ(norm_idx, my_real_idx);
        my_real_idx++;
      }
    }
  }
}
