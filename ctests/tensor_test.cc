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

// ====== Tensor fixture ======
class TensorFixtureTest : public testing::Test {
protected:
  TensorFixtureTest() {
    // t1
    t1->set({0, 0}, std::make_shared<Value>(1));
    t1->set({0, 1}, std::make_shared<Value>(2));
    t1->set({0, 2}, std::make_shared<Value>(3));
    t1->set({1, 0}, std::make_shared<Value>(4));
    t1->set({1, 1}, std::make_shared<Value>(5));
    t1->set({1, 2}, std::make_shared<Value>(6));

    // t2
    t2->set({0, 0}, std::make_shared<Value>(10));
    t2->set({0, 1}, std::make_shared<Value>(11));
    t2->set({1, 0}, std::make_shared<Value>(20));
    t2->set({1, 1}, std::make_shared<Value>(21));
    t2->set({2, 0}, std::make_shared<Value>(30));
    t2->set({2, 1}, std::make_shared<Value>(31));
  }

  // t1: [[1,2,3], [4,5,6]]
  // t2: [[10,11], [20,21], [30,31]]
  std::shared_ptr<Tensor> t1 = std::make_unique<Tensor>(std::vector<int>{2, 3});
  std::shared_ptr<Tensor> t2 = std::make_unique<Tensor>(std::vector<int>{3, 2});
};

TEST_F(TensorFixtureTest, AddTest) {
  // t3: [[140, 146], [320, 335]]
  std::shared_ptr<Tensor> t3 = std::make_unique<Tensor>(std::vector<int>{2, 3});
  // t1
    t3->set({0, 0}, std::make_shared<Value>(10));
    t3->set({0, 1}, std::make_shared<Value>(10));
    t3->set({0, 2}, std::make_shared<Value>(10));
    t3->set({1, 0}, std::make_shared<Value>(10));
    t3->set({1, 1}, std::make_shared<Value>(10));
    t3->set({1, 2}, std::make_shared<Value>(10));

  std::shared_ptr<Tensor> t_sum = t3->add(t1);

  EXPECT_EQ(t_sum->dims(), 2);

  EXPECT_EQ(t_sum->shape[0], 2);
  EXPECT_EQ(t_sum->shape[1], 3);

  EXPECT_DOUBLE_EQ(t_sum->get(0)->data, double(11));
  EXPECT_DOUBLE_EQ(t_sum->get(1)->data, double(12));
  EXPECT_DOUBLE_EQ(t_sum->get(2)->data, double(13));
  EXPECT_DOUBLE_EQ(t_sum->get(3)->data, double(14));
  EXPECT_DOUBLE_EQ(t_sum->get(4)->data, double(15));
  EXPECT_DOUBLE_EQ(t_sum->get(5)->data, double(16));
}

TEST_F(TensorFixtureTest, MatMulTest) {
  // t3: [[140, 146], [320, 335]]
  std::shared_ptr<Tensor> t3 = t1->matmul(t2);

  EXPECT_EQ(t3->dims(), 2);

  EXPECT_EQ(t3->shape[0], 2);
  EXPECT_EQ(t3->shape[1], 2);

  EXPECT_DOUBLE_EQ(t3->get(0)->data, double(140));
  EXPECT_DOUBLE_EQ(t3->get(1)->data, double(146));
  EXPECT_DOUBLE_EQ(t3->get(2)->data, double(320));
  EXPECT_DOUBLE_EQ(t3->get(3)->data, double(335));
}
