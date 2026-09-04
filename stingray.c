#include "stingray.h"
#include <math.h>
#include <stdlib.h>

#define PIXELS 784

// A group of stingrays are called fevers :) (this is essentially just the
// layer)
#define FEVER 4

Stingray *stingray_build(const int *pup_sizes, int n) {
  Stingray *stingray = (Stingray *)(malloc(sizeof(Stingray)));

  stingray->n_pups = FEVER;

  for (int i = 0; i < FEVER; i++) {
    Pup *pup = (Pup *)(malloc(sizeof(Pup)));

    // still need to generate random weights

    stingray->pups[i] = pup;
  }

  return stingray;
}

void free_stingray(Stingray *s) {
  free(s->pups);
  free(s);
}

void pup_forward(Pup *p, const float *in) {
  if (p == NULL) {
    return;
  }

  int n = sizeof(*in) / sizeof(float);

  // multiply weights and assign to the preact
  for (int i = 0; i < n; i++) {
    // z = Wx + b
    *(p->pre_act + i) = *(in) * *(p->w) + *(p->b);
  }

  // activation functions
  for (int i = 0; i < n; i++) {
    *(p->post_act + i) = 1 / (1 + expf(*(p->pre_act + i)));
  }
}

float *stingray_forward(Stingray *s, const float *in) {
  pup_forward(*(s->pups), in);
}
