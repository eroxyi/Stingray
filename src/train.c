#include "include/dataset.h"
#include "include/stingray.h"
#include "read.c"
#include "stingray.c"

int main() {
  Dataset *set = parse("../data/");

  int size[] = {784, 128, 64, 10};
  Stingray *net = stingray_build(size, 4);

  train(net, set, 10, 0.1f);

  free_dataset(set);
  free_stingray(net);

  return 0;
}
