#include <gtest/gtest.h>
#include "utils.h"

TEST(RandomNumberGenerator, IsConsistent) {
  int seed = 12345;

  // Create two instances of RandomNumberGenerator with the same seed
  RandomNumberGenerator rng1(seed);
  RandomNumberGenerator rng2(seed);

  // Generate a sequence of random numbers from both instances
  std::vector<double> sequence1;
  std::vector<double> sequence2;
  for (int i = 0; i < 10; ++i) {
    sequence1.push_back(rng1.generate());
    sequence2.push_back(rng2.generate());
  }

  // Check that both sequences are identical
  EXPECT_EQ(sequence1.size(), sequence2.size());
  for (size_t i = 0; i < sequence1.size(); ++i) {
    EXPECT_DOUBLE_EQ(sequence1[i], sequence2[i]);
  }
}
