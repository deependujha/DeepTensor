#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "loss.h"
#include "tensor.h"
#include "value.h"

TEST(LossTest, MSELossTest) {
  std::shared_ptr<Tensor> x = std::make_shared<Tensor>(std::vector<int>{3});
  std::shared_ptr<Tensor> y = std::make_shared<Tensor>(std::vector<int>{3});

  // set values
  x->set(0, std::make_shared<Value>(1.0));
  x->set(1, std::make_shared<Value>(2.0));
  x->set(2, std::make_shared<Value>(3.0));

  y->set(0, std::make_shared<Value>(1.2));
  y->set(1, std::make_shared<Value>(2.3));
  y->set(2, std::make_shared<Value>(3.2));

  std::shared_ptr<Value> out = mean_squared_error(x, y);

  double expectedValue = ((0.2 * 0.2) + (0.3 * 0.3) + (0.2 * 0.2)) / 3;

  EXPECT_DOUBLE_EQ(out->data, expectedValue);
}

TEST(LossTest, CrossEntropyLossTest) {
  std::shared_ptr<Tensor> x = std::make_shared<Tensor>(std::vector<int>{3});
  std::shared_ptr<Tensor> y = std::make_shared<Tensor>(std::vector<int>{3});

  // set values
  x->set(0, std::make_shared<Value>(2.5));
  x->set(1, std::make_shared<Value>(-3.7));
  x->set(2, std::make_shared<Value>(2.35));

  y->set(0, std::make_shared<Value>(1));
  y->set(1, std::make_shared<Value>(0));
  y->set(2, std::make_shared<Value>(0));

  std::shared_ptr<Value> out = cross_entropy(x, y);

  double expectedValue = 0.6220; // calculated using pytorch nn.CrossEntropyLoss
  double tolerance = 0.0001;    // Compare up to 4 decimal places


  EXPECT_NEAR(out->data, expectedValue, tolerance);
}
