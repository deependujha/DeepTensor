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
