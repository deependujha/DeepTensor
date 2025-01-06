#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "non_linear.h"
#include "value.h"

TEST(NonLinearValue, ReluTest) {
  std::shared_ptr<Value> v = std::make_shared<Value>(5);
  std::shared_ptr<Value> v_new = v->relu();

  EXPECT_DOUBLE_EQ(v->data, double(5));
  EXPECT_DOUBLE_EQ(v_new->data, double(5));
  EXPECT_DOUBLE_EQ(v->grad, double(0));
  EXPECT_DOUBLE_EQ(v_new->grad, double(0));
  v_new->backward();
  EXPECT_DOUBLE_EQ(v->grad, double(1));
  EXPECT_DOUBLE_EQ(v_new->grad, double(1));

  std::shared_ptr<Value> v_neg = std::make_shared<Value>(-5);
  std::shared_ptr<Value> v_neg_new = v_neg->relu();

  EXPECT_DOUBLE_EQ(v_neg->data, double(-5));
  EXPECT_DOUBLE_EQ(v_neg_new->data, double(0));
  EXPECT_DOUBLE_EQ(v_neg->grad, double(0));
  EXPECT_DOUBLE_EQ(v_neg_new->grad, double(0));
  v_neg_new->backward();
  EXPECT_DOUBLE_EQ(v_neg->grad, double(0));
  EXPECT_DOUBLE_EQ(v_neg_new->grad, double(1));
}

TEST(NonLinearValue, TanhTest) {
  std::shared_ptr<Value> v = std::make_shared<Value>(0.5);
  std::shared_ptr<Value> v_new = v->tanh();

  double tanhValue = std::tanh(0.5);

  EXPECT_DOUBLE_EQ(v->data, double(0.5));
  EXPECT_DOUBLE_EQ(v_new->data, double(tanhValue));
  EXPECT_DOUBLE_EQ(v->grad, double(0));
  EXPECT_DOUBLE_EQ(v_new->grad, double(0));
  v_new->backward();
  EXPECT_DOUBLE_EQ(v->grad, double(1 - (tanhValue * tanhValue)));
  EXPECT_DOUBLE_EQ(v_new->grad, double(1));
}

TEST(NonLinearValue, SigmoidTest) {
  std::shared_ptr<Value> v = std::make_shared<Value>(0.5);
  std::shared_ptr<Value> v_new = v->sigmoid();

  double sigmoidValue = 1 / (1 + std::exp(-0.5));

  EXPECT_DOUBLE_EQ(v->data, double(0.5));
  EXPECT_DOUBLE_EQ(v_new->data, double(sigmoidValue));
  EXPECT_DOUBLE_EQ(v->grad, double(0));
  EXPECT_DOUBLE_EQ(v_new->grad, double(0));
  v_new->backward();
  EXPECT_DOUBLE_EQ(v->grad, double(sigmoidValue * (1 - sigmoidValue)));
  EXPECT_DOUBLE_EQ(v_new->grad, double(1));
}

TEST(NonLinearValue, LeakyReluTest) {
  std::shared_ptr<Value> v = std::make_shared<Value>(5);
  std::shared_ptr<Value> v_new = v->leakyRelu(0.1);

  EXPECT_DOUBLE_EQ(v->data, double(5));
  EXPECT_DOUBLE_EQ(v_new->data, double(5));
  EXPECT_DOUBLE_EQ(v->grad, double(0));
  EXPECT_DOUBLE_EQ(v_new->grad, double(0));
  v_new->backward();
  EXPECT_DOUBLE_EQ(v->grad, double(1));
  EXPECT_DOUBLE_EQ(v_new->grad, double(1));

  std::shared_ptr<Value> v_neg = std::make_shared<Value>(-5);
  std::shared_ptr<Value> v_neg_new = v_neg->leakyRelu(0.1);

  EXPECT_DOUBLE_EQ(v_neg->data, double(-5));
  EXPECT_DOUBLE_EQ(v_neg_new->data, double(-0.5));
  EXPECT_DOUBLE_EQ(v_neg->grad, double(0));
  EXPECT_DOUBLE_EQ(v_neg_new->grad, double(0));
  v_neg_new->backward();
  EXPECT_DOUBLE_EQ(v_neg->grad, double(0.1));
  EXPECT_DOUBLE_EQ(v_neg_new->grad, double(1));
}

/// chatgpt wrote this grad computation for gelu function
double gelu_grad(double x) {
  const double k = std::sqrt(2.0 / M_PI);
  const double c = 0.044715;
  double tanh_input = k * (x + c * x * x * x);
  double tanh_val = std::tanh(tanh_input);
  double sech2 = 1 - tanh_val * tanh_val; // sech^2(x) = 1 - tanh^2(x)
  double grad =
      0.5 * tanh_val + (0.5 * x * sech2 * k * (1 + 3 * c * x * x)) + 0.5;
  return grad;
}

TEST(NonLinearValue, GeluTest) {
  std::shared_ptr<Value> v = std::make_shared<Value>(0.5);
  std::shared_ptr<Value> v_new = v->gelu();

  const double sqrt2OverPi = std::sqrt(2.0 / M_PI);
  double tanhInternalValue = (0.5 + 0.044715 * (0.5 * 0.5 * 0.5));
  double geluValue =
      0.5 * 0.5 * (1 + std::tanh(sqrt2OverPi * tanhInternalValue));

  EXPECT_DOUBLE_EQ(v->data, double(0.5));
  EXPECT_DOUBLE_EQ(v_new->data, double(geluValue));
  EXPECT_DOUBLE_EQ(v->grad, double(0));
  EXPECT_DOUBLE_EQ(v_new->grad, double(0));
  v_new->backward();

  EXPECT_DOUBLE_EQ(v->grad, double(gelu_grad(v->data)));
  EXPECT_DOUBLE_EQ(v_new->grad, double(1));
}

// ========= function non-linear activations =========

TEST(FunctionalNonLinear, ReluTest) {
  std::vector<std::shared_ptr<Value>> inputs = {
      std::make_shared<Value>(-1.0),
      std::make_shared<Value>(0.0),
      std::make_shared<Value>(2.0)};
  std::vector<std::shared_ptr<Value>> outputs = NonLinear::relu(inputs);

  ASSERT_EQ(inputs.size(), outputs.size());
  EXPECT_DOUBLE_EQ(outputs[0]->data, 0.0);
  EXPECT_DOUBLE_EQ(outputs[1]->data, 0.0);
  EXPECT_DOUBLE_EQ(outputs[2]->data, 2.0);

  // Backward pass
  outputs[0]->backward();
  EXPECT_DOUBLE_EQ(inputs[0]->grad, 0.0);
  outputs[2]->backward();
  EXPECT_DOUBLE_EQ(inputs[2]->grad, 1.0);
}

TEST(FunctionalNonLinear, TanhTest) {
  std::vector<std::shared_ptr<Value>> inputs = {
      std::make_shared<Value>(-1.0),
      std::make_shared<Value>(0.0),
      std::make_shared<Value>(1.0)};
  std::vector<std::shared_ptr<Value>> outputs = NonLinear::tanh(inputs);

  ASSERT_EQ(inputs.size(), outputs.size());
  EXPECT_DOUBLE_EQ(outputs[0]->data, std::tanh(-1.0));
  EXPECT_DOUBLE_EQ(outputs[1]->data, std::tanh(0.0));
  EXPECT_DOUBLE_EQ(outputs[2]->data, std::tanh(1.0));

  // Backward pass
  outputs[1]->backward();
  EXPECT_DOUBLE_EQ(inputs[1]->grad, 1 - std::pow(std::tanh(0.0), 2));
}

TEST(FunctionalNonLinear, SigmoidTest) {
  std::vector<std::shared_ptr<Value>> inputs = {
      std::make_shared<Value>(-1.0),
      std::make_shared<Value>(0.0),
      std::make_shared<Value>(1.0)};
  std::vector<std::shared_ptr<Value>> outputs = NonLinear::sigmoid(inputs);

  ASSERT_EQ(inputs.size(), outputs.size());
  EXPECT_DOUBLE_EQ(outputs[0]->data, 1 / (1 + std::exp(1.0)));
  EXPECT_DOUBLE_EQ(outputs[1]->data, 0.5);
  EXPECT_DOUBLE_EQ(outputs[2]->data, 1 / (1 + std::exp(-1.0)));

  // Backward pass
  outputs[1]->backward();
  EXPECT_DOUBLE_EQ(inputs[1]->grad, 0.25); // Sigmoid'(0.0) = 0.5 * 0.5
}

TEST(FunctionalNonLinear, LeakyReluTest) {
  double alpha = 0.1;
  std::vector<std::shared_ptr<Value>> inputs = {
      std::make_shared<Value>(-1.0),
      std::make_shared<Value>(0.0),
      std::make_shared<Value>(2.0)};
  std::vector<std::shared_ptr<Value>> outputs =
      NonLinear::leakyRelu(alpha, inputs);

  ASSERT_EQ(inputs.size(), outputs.size());
  EXPECT_DOUBLE_EQ(outputs[0]->data, -0.1);
  EXPECT_DOUBLE_EQ(outputs[1]->data, 0.0);
  EXPECT_DOUBLE_EQ(outputs[2]->data, 2.0);

  // Backward pass
  outputs[0]->backward();
  EXPECT_DOUBLE_EQ(
      inputs[0]->grad, alpha); // Gradient = alpha for negative values
}

TEST(FunctionalNonLinear, SoftmaxTest) {
  std::vector<std::shared_ptr<Value>> inputs{
      std::make_shared<Value>(1.0),
      std::make_shared<Value>(2.0),
      std::make_shared<Value>(3.0)};
  std::vector<std::shared_ptr<Value>> outputs = NonLinear::softmax(inputs);

    ASSERT_EQ(inputs.size(), outputs.size());

    double sum = std::exp(1.0) + std::exp(2.0) + std::exp(3.0);
    EXPECT_NEAR(outputs[0]->data, std::exp(1.0) / sum, 1e-5);
    EXPECT_NEAR(outputs[1]->data, std::exp(2.0) / sum, 1e-5);
    EXPECT_NEAR(outputs[2]->data, std::exp(3.0) / sum, 1e-5);

    // Backward pass
    outputs[1]->backward();
    EXPECT_NE(inputs[0]->grad, double(0));
}
