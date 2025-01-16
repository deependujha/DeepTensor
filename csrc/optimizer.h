#pragma once
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include "neural_network.h"

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step() = 0;
};

// stochastic gradient descent
class SGD : public Optimizer {
  std::shared_ptr<Model> m;

public:
  double learning_rate = 0.1;

  explicit SGD(std::shared_ptr<Model> m, double learning_rate)
      : m(std::move(m)), learning_rate(learning_rate) {}

  void step() override {
    const auto& m_para = this->m->parameters();
    for (auto& e : m_para) {
      e->data = e->data - this->learning_rate * e->grad;
    }
  }
};

// SGD with Momentum
class Momentum : public Optimizer {
  std::shared_ptr<Model> m;
  std::vector<double> velocity;

public:
  double learning_rate = 0.1;
  double momentum = 0.1;

  explicit Momentum(
      std::shared_ptr<Model> m,
      double learning_rate,
      double momentum)
      : m(std::move(m)), learning_rate(learning_rate), momentum(momentum) {
    velocity.resize(m->parameters().size(), 0);
  }

  void step() override {
    const auto& m_para = this->m->parameters();
    for (int i = 0; i < m_para.size(); i++) {
      velocity[i] = this->momentum * velocity[i] + m_para[i]->grad;
      m_para[i]->data = m_para[i]->data - this->learning_rate * velocity[i];
    }
  }
};

// Nesterov Accelerated Gradient (NAG) - we need to compute gradient at
// look_ahead position, which is quite complex for now. Let's move forward.
// class NAG : public Optimizer {
//   std::shared_ptr<Model> m;
//   std::vector<double> velocity;

// public:
//   double learning_rate = 0.1;
//   double momentum = 0.1;
//   double gamma = 0.9;

//   explicit NAG(
//       std::shared_ptr<Model> m,
//       double learning_rate,
//       double momentum,
//       double gamma)
//       : m(std::move(m)), learning_rate(learning_rate), momentum(momentum),
//       gamma(gamma) {
//     velocity.resize(m->parameters().size(), 0);
//   }

//   void step() override {
//     const auto& m_para = this->m->parameters();
//     for (int i = 0; i < m_para.size(); i++) {
//         double velocity_look_ahead = this->gamma * velocity[i];
//         std::shared_ptr<Value> m_para_temp =
//         m_para[i]->sub(velocity_look_ahead); m_para[i]->data = m_para_temp -
//       velocity[i] = this->momentum * velocity[i] + m_para[i]->grad;
//       m_para[i]->data = m_para[i]->data - this->learning_rate * velocity[i];
//     }
//   }
// };

// AdaGrad (Adaptive Gradient Algorithm) - great for sparse datasets
class AdaGrad : public Optimizer {
  std::shared_ptr<Model> m;
  std::vector<double> prev_grad_square;
  double epsilon = 0.000001;

public:
  double learning_rate = 0.1;

  explicit AdaGrad(std::shared_ptr<Model> m, double learning_rate)
      : m(std::move(m)), learning_rate(learning_rate) {
    prev_grad_square.resize(m->parameters().size(), 0);
  }

  void step() override {
    const auto& m_para = this->m->parameters();
    for (int i = 0; i < m_para.size(); i++) {
      prev_grad_square[i] =
          prev_grad_square[i] + (m_para[i]->grad * m_para[i]->grad);
      m_para[i]->data = m_para[i]->data -
          (this->learning_rate * m_para[i]->grad) /
              std::sqrt(prev_grad_square[i] + this->epsilon);
    }
  }
};

// RMSProp (Root Mean Square Propagation)
class RMSprop : public Optimizer {
  std::shared_ptr<Model> m;
  std::vector<double> prev_grad_square;
  double epsilon = 1e-8;
  double beta = 0.9; // Decay rate for the moving average

public:
  double learning_rate = 0.001;

  explicit RMSprop(
      std::shared_ptr<Model> m,
      double learning_rate,
      double beta = 0.9)
      : m(std::move(m)), learning_rate(learning_rate), beta(beta) {
    prev_grad_square.resize(m->parameters().size(), 0.0);
  }

  void step() override {
    const auto& m_para = this->m->parameters();
    for (size_t i = 0; i < m_para.size(); i++) {
      // Update moving average of squared gradients
      prev_grad_square[i] = beta * prev_grad_square[i] +
          (1 - beta) * (m_para[i]->grad * m_para[i]->grad);

      // Update parameter
      m_para[i]->data = m_para[i]->data -
          (learning_rate * m_para[i]->grad) /
              std::sqrt(prev_grad_square[i] + epsilon);
    }
  }
};

// ADAM (Adaptive Moment Estimation)
//      Uses both: Momentum and adaptive gradient
class Adam : public Optimizer {
  std::shared_ptr<Model> m;
  std::vector<double> prev_grad_square;
  std::vector<double> velocity;
  double epsilon = 1e-8;
  double beta1 = 0.9; // Decay rate for the moving average
  double beta2 = 0.999; // Decay rate for the moving average
  int time = 1;

  void _initialize() {
    this->prev_grad_square.resize(m->parameters().size(), 0.0);
    this->velocity.resize(m->parameters().size(), 0.0);
  }

public:
  double learning_rate = 0.001;

  explicit Adam(
      std::shared_ptr<Model> m,
      double learning_rate,
      double beta1,
      double beta2)
      : m(std::move(m)),
        learning_rate(learning_rate),
        beta1(beta1),
        beta2(beta2) {
    _initialize();
  }
  explicit Adam(std::shared_ptr<Model> m, double learning_rate)
      : m(std::move(m)), learning_rate(learning_rate) {
    _initialize();
  }

  void step() override {
    const auto& m_para = this->m->parameters();

    double bias_correction1 = 1 - pow(beta1, time);
    double bias_correction2 = 1 - pow(beta2, time);

    for (size_t i = 0; i < m_para.size(); i++) {
      // Update moving average of velocity
      velocity[i] = beta1 * velocity[i] + (1 - beta1) * (m_para[i]->grad);
      // Update moving average of squared gradients
      prev_grad_square[i] = beta2 * prev_grad_square[i] +
          (1 - beta2) * (m_para[i]->grad * m_para[i]->grad);

      //   perform bias correction (to fix bias introduced introduced at t=0 due
      //   to taking v=0 and sq_grad = 0)
      double corrected_velocity = velocity[i] / bias_correction1;
      double corrected_grad_square = prev_grad_square[i] / bias_correction2;

      // Update parameter
      m_para[i]->data = m_para[i]->data -
          (learning_rate * corrected_velocity) /
              std::sqrt(corrected_grad_square + epsilon);
    }
    this->time++;
  }
};
