#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "layers/feed_forward_layer.h"
#include "layers/non_linear_layer.h"
#include "neural_network.h"

// it'll fail due to seed on linux generate different values

// TEST(ModelTest, FeedForward) {
//   // [
//   //    [
//   //        Value(data=0.364466, grad=0.000000),
//   //        Value(data=-0.389075, grad=0.000000)
//   //     ],
//   //    [
//   //        Value(data=0.967640, grad=0.000000),
//   //        Value(data=0.335070, grad=0.000000)
//   //     ]
//   // ]
//   std::vector<double> expected_feedforward_weights = {
//       0.364466,
//       -0.389075,
//       0.967640,
//       0.335070,
//       0,
//       0}; // 4th and 5th elements are two biases initialized with 0

//   double tolerance = 0.0001; // Compare up to 4 decimal places

//   int seed = 42;
//   std::shared_ptr<Model> model = std::make_shared<Model>(
//       std::vector<std::shared_ptr<Layer>>{
//           std::make_shared<FeedForwardLayer>(2, 2, seed, "XAVIER", "NORMAL"),
//       },
//       false);

//   int i = 0;

//   for (auto& e : model->parameters()) {
//     EXPECT_NEAR(e->data, expected_feedforward_weights[i], tolerance);
//     i++;
//   }
//   std::shared_ptr<Tensor> inp = std::make_shared<Tensor>(std::vector<int>{2});
//   inp->set(0, std::make_shared<Value>(0.5));
//   inp->set(1, std::make_shared<Value>(0.3));

//   // expected output:
//   // [0.472525, -0.0940165]
//   std::shared_ptr<Tensor> out = model->call(inp);

//   EXPECT_EQ(out->dims(), 1);
//   EXPECT_EQ(out->shape.size(), 1);
//   EXPECT_EQ(out->shape[0], 2);
//   EXPECT_NEAR(out->get(0)->data, 0.472525, tolerance);
//   EXPECT_NEAR(out->get(1)->data, -0.0940165, tolerance);

//   out->backward();

//   // expected grad: 0.5, 0.5, 0.3, 0.3, 1, 1
//   std::vector<double> expected_grad = {0.5, 0.5, 0.3, 0.3, 1, 1};

//   int idx = 0;
//   for (auto& e : model->parameters()) {
//     EXPECT_NEAR(e->grad, expected_grad[idx], tolerance);
//     idx++;
//   }
// }

// TEST(ModelTest, FeedForwardWithRelu) {
//   // [
//   //    [
//   //        Value(data=0.364466, grad=0.000000),
//   //        Value(data=-0.389075, grad=0.000000)
//   //     ],
//   //    [
//   //        Value(data=0.967640, grad=0.000000),
//   //        Value(data=0.335070, grad=0.000000)
//   //     ]
//   // ]
//   std::vector<double> expected_feedforward_weights = {
//       0.364466,
//       -0.389075,
//       0.967640,
//       0.335070,
//       0,
//       0}; // 4th and 5th elements are two biases initialized with 0

//   double tolerance = 0.0001; // Compare up to 4 decimal places

//   int seed = 42;
//   std::shared_ptr<Model> model = std::make_shared<Model>(
//       std::vector<std::shared_ptr<Layer>>{
//           std::make_shared<FeedForwardLayer>(2, 2, seed, "XAVIER", "NORMAL"),
//           std::make_shared<ReLu>(),
//       },
//       false);

//   int i = 0;

//   for (auto& e : model->parameters()) {
//     EXPECT_NEAR(e->data, expected_feedforward_weights[i], tolerance);
//     i++;
//   }
//   std::shared_ptr<Tensor> inp = std::make_shared<Tensor>(std::vector<int>{2});
//   inp->set(0, std::make_shared<Value>(0.5));
//   inp->set(1, std::make_shared<Value>(0.3));

//   // expected output:
//   // [0.472525, -0.0940165] ===Relu===> [0.472525, 0]
//   std::shared_ptr<Tensor> out = model->call(inp);

//   EXPECT_EQ(out->dims(), 1);
//   EXPECT_EQ(out->shape.size(), 1);
//   EXPECT_EQ(out->shape[0], 2);
//   EXPECT_NEAR(out->get(0)->data, 0.472525, tolerance);
//   EXPECT_NEAR(out->get(1)->data, 0, tolerance);

//   out->backward();

//   // expected grad: 0.5, 0, 0.3, 0, 1, 0 # relu causes grad to be 0 for negative
//   // output
//   std::vector<double> expected_grad = {0.5, 0, 0.3, 0, 1, 0};

//   int idx = 0;
//   for (auto& e : model->parameters()) {
//     EXPECT_NEAR(e->grad, expected_grad[idx], tolerance);
//     idx++;
//   }
// }
