#include <gtest/gtest.h>
#include <memory>
#include "neural_network.h"
#include "value.h"

TEST(NeuralNetwork, NeuronTest) {
  std::shared_ptr<Neuron> n =
      std::make_shared<Neuron>(3, true, 42); // takes 5 inputs

  std::vector<std::shared_ptr<Value>> input = {
      std::make_shared<Value>(1.5),
      std::make_shared<Value>(2.5),
      std::make_shared<Value>(3.0)};

    std::shared_ptr<Value> out = n->call(input);

    EXPECT_DOUBLE_EQ(out->grad, 0);

    out->backward();

    EXPECT_DOUBLE_EQ(out->grad, 10);
}
