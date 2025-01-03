#include <gtest/gtest.h>
#include "value.h"

// `Value fixture` for googletest
class ValueTest : public testing::Test {
protected:
  ValueTest() {
    this->v1 = new Value(5.1);
    this->v2 = new Value(4.2);
  }

  //   ~ValueTest() override = default;

  Value* v1 = new Value(5.1);
  Value* v2 = new Value(4.2);
};

// ========= TESTs =========

TEST_F(ValueTest, ValueAddition) {
  Value* v3 = v1->add(v2);

  EXPECT_DOUBLE_EQ(v1->data, double(5.1));
  EXPECT_DOUBLE_EQ(v2->data, double(4.2));
  EXPECT_DOUBLE_EQ(v3->data, double(9.3));

  v3->grad = -2.3;

  EXPECT_DOUBLE_EQ(v1->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v2->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  v3->executeBackWardMethod();

  EXPECT_DOUBLE_EQ(v1->grad, double(-2.3));
  EXPECT_DOUBLE_EQ(v2->grad, double(-2.3));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  delete v3;
}

TEST_F(ValueTest, ValueSubtraction) {
  Value* v3 = v1->sub(v2);

  EXPECT_DOUBLE_EQ(v1->data, double(5.1));
  EXPECT_DOUBLE_EQ(v2->data, double(4.2));
  EXPECT_DOUBLE_EQ(v3->data, double(5.1 - 4.2));

  v3->grad = -2.3;

  EXPECT_DOUBLE_EQ(v1->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v2->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  v3->executeBackWardMethod();

  EXPECT_DOUBLE_EQ(v1->grad, double(-2.3));
  EXPECT_DOUBLE_EQ(v2->grad, double(2.3));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  delete v3;
}

TEST_F(ValueTest, ValueMultiplication) {
  Value* v3 = v1->mul(v2);

  EXPECT_DOUBLE_EQ(v1->data, double(5.1));
  EXPECT_DOUBLE_EQ(v2->data, double(4.2));
  EXPECT_DOUBLE_EQ(v3->data, double(5.1 * 4.2));

  v3->grad = -2.3;

  EXPECT_DOUBLE_EQ(v1->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v2->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  v3->executeBackWardMethod();

  // differentiation of f(x,y) = x * y => x+y
  EXPECT_DOUBLE_EQ(v1->grad, double(-2.3 * v2->data));
  EXPECT_DOUBLE_EQ(v2->grad, double(-2.3 * v1->data));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  delete v3;
}

TEST_F(ValueTest, ValuePower) {
  Value* v3 = v1->pow(3);

  EXPECT_DOUBLE_EQ(v1->data, double(5.1));
  EXPECT_DOUBLE_EQ(v3->data, double(5.1 * 5.1 * 5.1));

  v3->grad = -2.3;

  EXPECT_DOUBLE_EQ(v1->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  v3->executeBackWardMethod();

  // differentiation of f(x,y) = x * y => x+y
  EXPECT_DOUBLE_EQ(v1->grad, double(-2.3 * 3 * v1->data * v1->data));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  delete v3;
}

TEST_F(ValueTest, ValueRelu) {
  Value* v3 = v1->relu();

  EXPECT_DOUBLE_EQ(v1->data, double(5.1));
  EXPECT_DOUBLE_EQ(v3->data, double(5.1));

  v3->grad = -2.3;

  EXPECT_DOUBLE_EQ(v1->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  v3->executeBackWardMethod();

  // differentiation of f(x,y) = x * y => x+y
  EXPECT_DOUBLE_EQ(v1->grad, double(-2.3));
  EXPECT_DOUBLE_EQ(v3->grad, double(-2.3));

  delete v3;

  // ======= relu second case =======

  v3 = new Value(-5.2);
  Value* v4 = v3->relu();

  EXPECT_DOUBLE_EQ(v3->data, double(-5.2));
  EXPECT_DOUBLE_EQ(v4->data, double(0));

  v4->grad = -2.3;

  EXPECT_DOUBLE_EQ(v3->grad, double(0.0));
  EXPECT_DOUBLE_EQ(v4->grad, double(-2.3));

  v4->executeBackWardMethod();

  // differentiation of f(x,y) = x * y => x+y
  EXPECT_DOUBLE_EQ(v3->grad, double(0));
  EXPECT_DOUBLE_EQ(v4->grad, double(-2.3));

  delete v3;
  delete v4;
}

TEST_F(ValueTest, CombinedTest) {
  // f(y) = x**2 + 5*y - 1
  Value* x = new Value(7.0);
  Value* y = new Value(2.0);

  Value* z = x->pow(2)->add(y->mul(5))->sub(1);

  EXPECT_DOUBLE_EQ(x->data, double(7.0));
  EXPECT_DOUBLE_EQ(y->data, double(2.0));
  EXPECT_DOUBLE_EQ(z->data, double(58.0));

  z->grad = 1.0;
  // x->grad = (2 * x) * z_grad
  // y->grad = (5) * z_grad

  EXPECT_DOUBLE_EQ(x->grad, double(0.0));
  EXPECT_DOUBLE_EQ(y->grad, double(0.0));
  EXPECT_DOUBLE_EQ(z->grad, double(1.0));

  z->backward();

  EXPECT_DOUBLE_EQ(x->grad, double(2 * 7.0 * 1.0));
  EXPECT_DOUBLE_EQ(y->grad, double(5 * 1.0));
  EXPECT_DOUBLE_EQ(z->grad, double(1.0));
}
