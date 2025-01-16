#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include "constant.h"

class RandomNumberGenerator {
public:
  // Constructor with initialization technique, mode, input size, output size,
  // and optional seed
  RandomNumberGenerator(
      const std::string& technique,
      const std::string& mode,
      int input_size,
      int output_size,
      int seed)
      : gen(seed) {
    // Seed defaults to a random value if not provided

    if (technique != constant::HE && technique != constant::XAVIER) {
      throw std::runtime_error(
          "RandomNumberGenerator expects 'technique' to be either 'XAVIER' or 'HE'. Got: " +
          technique);
    }
    if (mode != constant::UNIFORM && mode != constant::NORMAL) {
      throw std::runtime_error(
          "RandomNumberGenerator expects 'mode' to be either 'UNIFORM' or 'NORMAL'. Got: " +
          mode);
    }

    this->_initializer(technique, mode, input_size, output_size);
    this->mode = mode;
  }

  // Method to generate a random number based on the chosen mode
  double generate() {
    if (this->mode == constant::UNIFORM) {
      return uniform_dis(gen);
    } else if (this->mode == constant::NORMAL) {
      return normal_dis(gen);
    }
    throw std::runtime_error("Invalid mode encountered during generation.");
  }

private:
  std::mt19937 gen; // Mersenne Twister random number generator
  std::uniform_real_distribution<double> uniform_dis; // Uniform distribution
  std::normal_distribution<double> normal_dis; // Normal distribution
  std::string mode; // Selected mode (UNIFORM or NORMAL)

  void _initializer(
      const std::string& technique,
      const std::string& mode,
      int input_size,
      int output_size) {
    double variance = 0.0;
    if (technique == constant::XAVIER) {
      variance = (mode == constant::NORMAL)
          ? std::sqrt(double(2) / (input_size + output_size))
          : std::sqrt(double(6) / (input_size + output_size));
    } else if (technique == constant::HE) {
      variance = (mode == constant::NORMAL) ? std::sqrt(double(2) / input_size)
                                            : std::sqrt(double(6) / input_size);
    } else {
      throw std::runtime_error(
          "Should not have happened. Expected technique: XAVIER | HE. Got: " +
          technique);
    }

    if (mode == constant::NORMAL) {
      this->normal_dis = std::normal_distribution<double>(0.0, variance);
    } else {
      this->uniform_dis =
          std::uniform_real_distribution<double>(-variance, variance);
    }
  }
};
