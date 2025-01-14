#include "loss.h"
#include <memory>
#include <stdexcept>
#include "value.h"

std::shared_ptr<Value> mean_squared_error(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> y) {
  if (x->shape != y->shape) {
    std::string x_shape_str = x->tensor_shape_str();
    std::string y_shape_str = y->tensor_shape_str();
    std::string error_string =
        "Shapes of the two tensors for computing MSE don't match: tensor-1 shape (" +
        x_shape_str + ") vs tensor-1 shape(" + y_shape_str + ")\n";
    throw std::runtime_error(error_string);
  }
  std::shared_ptr<Value> out = std::make_shared<Value>(0.0);
  int n = x->maxIdx;

  for (int i = 0; i <= n; i++) {
    std::shared_ptr<Value> diff = x->get(i)->sub(y->get(i));
    std::shared_ptr<Value> diff_squared = diff->pow(2);
    out = out->add(diff_squared);
  }
  return out->div(n + 1);
}

std::shared_ptr<Value> cross_entropy(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> actual){
    if (logits->shape != actual->shape) {
    std::string x_shape_str = logits->tensor_shape_str();
    std::string y_shape_str = actual->tensor_shape_str();
    std::string error_string =
        "Shapes of the two tensors for computing cross_entropy don't match: tensor-1 shape (" +
        x_shape_str + ") vs tensor-1 shape(" + y_shape_str + ")\n";
    throw std::runtime_error(error_string);
  }
    // compute softmax of logits
    std::shared_ptr<Value> out = std::make_shared<Value>(0.0);
    std::shared_ptr<Tensor> logits_softmax = logits->softmax();

    int n = logits->maxIdx;
    for(int i=0;i<=n;i++){
        std::shared_ptr logits_ln = logits_softmax->get(i)->ln();
        std::shared_ptr<Value> pro_log = actual->get(i)->mul(logits_ln); // product of log

        out = out->add(pro_log);
    }

    return out->mul(-1); // not averaging it
}
