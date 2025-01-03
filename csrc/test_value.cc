#include <iostream>
#include "value.h"

int main() {
  Value* v1 = new Value(5.1);
  Value* v2 = new Value(4.2);

  Value* v3 = v1->add(v2);
  std::cout << "value v1: " << v1->data << "\n";
  std::cout << "value v2: " << v2->data << "\n";
  std::cout << "value v3: " << v3->data << "\n";

  v3->grad = -2.3;
  std::cout << "Grad v1: " << v1->grad << "\n";
  std::cout << "Grad v2: " << v2->grad << "\n";
  std::cout << "Grad v3: " << v3->grad << "\n";
  v3->executeBackWardMethod();

  std::cout << "Grad v1: " << v1->grad << "\n";
  std::cout << "Grad v2: " << v2->grad << "\n";
  std::cout << "Grad v3: " << v3->grad << "\n";
}
