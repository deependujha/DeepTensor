#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "layers/feed_forward_layer.h"
#include "layers/non_linear_layer.h"
#include "neural_network.h"

TEST(ModelTest, IsWorking) {
  std::shared_ptr<Model> model = std::make_shared<Model>(
      std::vector<std::shared_ptr<Layer>>{
          std::make_shared<FeedForwardLayer>(2, 8),
          std::make_shared<Sigmoid>(),
          std::make_shared<FeedForwardLayer>(8, 1),
          std::make_shared<Tanh>()},
      false);
std::cerr<<"model creation done\n";
  std::shared_ptr<Tensor> inp = std::make_shared<Tensor>(std::vector<int>{2});
  inp->set(0, std::make_shared<Value>(5));
  inp->set(1, std::make_shared<Value>(2));
std::cerr<<"model input done\n";

  std::shared_ptr<Tensor> out = model->call(inp);
std::cerr<<"model output done\n";

  EXPECT_EQ(out->dims(), 1);
  EXPECT_NE(out->get(0)->data, 0);

  out->backward();

  EXPECT_DOUBLE_EQ(out->get(0)->grad, 1);
}
