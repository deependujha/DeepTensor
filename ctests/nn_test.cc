#include <gtest/gtest.h>
#include <memory>
#include "neural_network.h"
#include "value.h"

TEST(NeuralNetworkTest, NeuronUsage) {
  std::shared_ptr<Neuron> n =
      std::make_shared<Neuron>(3, true, 42); // takes 5 inputs

  std::vector<std::shared_ptr<Value>> input = {
      std::make_shared<Value>(1.5),
      std::make_shared<Value>(2.5),
      std::make_shared<Value>(3.0)};

  std::shared_ptr<Value> out = n->call(input);

  EXPECT_DOUBLE_EQ(out->grad, 0);

  out->backward();

  EXPECT_DOUBLE_EQ(out->grad, 1);
}

// Test for the Neuron class
TEST(NeuronTest, ParametersInitialization) {
    Neuron n(3, true, 42);  // 3 inputs, non-linear, seed = 42

    auto params = n.parameters();
    EXPECT_EQ(params.size(), 4); // 3 weights + 1 bias
    for (const auto& p : params) {
        EXPECT_NE(p, nullptr);
    }
}

TEST(NeuronTest, ForwardPass) {
    Neuron n(2, true, 42);

    auto v1 = std::make_shared<Value>(1.0);
    auto v2 = std::make_shared<Value>(2.0);

    auto result = n.call({v1, v2});
    EXPECT_NE(result, nullptr);
}

// Test for the Layer class
TEST(LayerTest, Initialization) {
    Layer l(3, 2, true, 42);  // 3 inputs, 2 outputs, non-linear, seed = 42

    auto params = l.parameters();
    EXPECT_EQ(params.size(), 8); // (3 weights + 1 bias) * 2 neurons
}

TEST(LayerTest, ForwardPass) {
    Layer l(2, 3, true, 42);

    auto v1 = std::make_shared<Value>(1.0);
    auto v2 = std::make_shared<Value>(2.0);

    auto outputs = l.call({v1, v2});
    EXPECT_EQ(outputs.size(), 3);
    for (const auto& out : outputs) {
        EXPECT_NE(out, nullptr);
    }
}

// Test for the MLP class
TEST(MLPTest, Initialization) {
    MLP mlp(3, {4, 2}, true, 42);  // 3 inputs, [4, 2] layer sizes, non-linear, seed = 42

    auto params = mlp.parameters();
    EXPECT_EQ(params.size(), (3 * 4 + 4) + (4 * 2 + 2)); // weights and biases
}

TEST(MLPTest, ForwardPass) {
    MLP mlp(2, {3, 2}, true, 42);

    auto v1 = std::make_shared<Value>(1.0);
    auto v2 = std::make_shared<Value>(2.0);

    auto outputs = mlp.call({v1, v2});
    EXPECT_EQ(outputs.size(), 2);
    for (const auto& out : outputs) {
        EXPECT_NE(out, nullptr);
    }
}
