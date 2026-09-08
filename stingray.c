#include "stingray.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define PIXELS 784

// A group of stingrays are called fevers :) (this is essentially just the
// layer)
#define FEVER 4
#define FEVER_SIZE 100
#define OUTPUT_NODES 10

Stingray *stingray_build(const int *pup_sizes, int n) {
  srand(time(NULL));

  Stingray *stingray = (Stingray *)(malloc(sizeof(Stingray)));
  stingray->pups = (Pup *)(malloc(sizeof(Pup) * (n - 1)));
  stingray->n_pups = n - 1;

  for (int i = 0; i < n - 1; i++) {
    Pup *pup = &stingray->pups[i];

    pup->n_in = pup_sizes[i];
    pup->n_out = pup_sizes[i + 1];

    int numW = pup_sizes[i] * pup_sizes[i + 1];
    pup->w = (float *)malloc(sizeof(float) * (numW));
    pup->b = (float *)malloc(sizeof(float) * (pup_sizes[i + 1]));

    pup->pre_act = (float *)malloc(sizeof(float) * pup_sizes[i + 1]);
    pup->post_act = (float *)malloc(sizeof(float) * pup_sizes[i + 1]);
    pup->error = (float *)malloc(sizeof(float) * pup_sizes[i + 1]);

    // randomizing weights and setting biases to zero
    for (int j = 0; j < numW; j++) {
      double scale = (double)rand() / RAND_MAX;
      *((pup->w) + j) = ((-1 + scale) * 2);
    }

    for (int k = 0; k < pup_sizes[i + 1]; k++) {
      *((pup->b) + k) = 0;
    }
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

  p++;
  pup_forward(p, p->post_act);
}

float *stingray_forward(Stingray *s, const float *in) {
  pup_forward((s->pups), in);
}

// backpropagation
void compute_error(Stingray *s, const float *target) {
  Pup *out = &s->pups[s->n_pups - 1];
  int size = out->n_out;

  for (int i = 0; i < size; i++) {
    out->error[i] = (out->post_act[i] - target[i]) * out->post_act[i] *
                    (1 - out->post_act[i]);
  }
}

void pup_backward(Pup *p, Pup *next, const float *input) {
  float dL_da;
  for (int i = 0; i < p->n_out; i++) {
    dL_da = 0;
    for (int j = 0; j < next->n_out; j++) {
      // sum of error * weight, chain rule
      dL_da += next->error[j] * next->w[j * (p->n_out) + i];
    }
    p->error[i] = dL_da * (p->post_act[i]) * (1 - (p->post_act[i]));
  }
}

void back_prop(Stingray *s, const float *input, const float *target,
               float rate) {
  stingray_forward(s, input);
  compute_error(s, target);

  for (int i = s->n_pups - 2; i >= 0; i--) {
    const float *input_to_l = (i == 0) ? input : s->pups[i - 1].post_act;
    pup_backward(&s->pups[i], &s->pups[i + 1], input_to_l);
  }
}

// saving the actual file
