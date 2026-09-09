#include "stingray.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

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
  for (int i = 0; i < s->n_pups; i++) {
    free(s->pups[i].w);
    free(s->pups[i].b);
    free(s->pups[i].pre_act);
    free(s->pups[i].post_act);
    free(s->pups[i].error);
  }

  free(s->pups);
  free(s);
}

void pup_forward(Pup *p, const float *in) {
  for (int i = 0; i < p->n_out; i++) {
    // loop through the weights
    p->pre_act[i] = 0;
    for (int j = 0; j < p->n_in; j++) {
      p->pre_act[i] += ((p->w[i * p->n_in + j] * in[j]));
    }
    p->pre_act[i] += p->b[i];
    (p->post_act[i]) = (1.0 / (1 + expf(-p->pre_act[i])));
  }
}

float *stingray_forward(Stingray *s, const float *in) {
  pup_forward(s->pups, in);

  for (int i = 1; i < s->n_pups; i++) {
    pup_forward((s->pups + i), (s->pups + i - 1)->post_act);
  }

  Pup *target = &(s->pups[s->n_pups - 1]);

  return target->post_act;
}

// backpropagation
void compute_error(Stingray *s, const float *target, float rate) {
  Pup *out = &s->pups[s->n_pups - 1];
  Pup *prev = &s->pups[s->n_pups - 2];

  int size = out->n_out;
  for (int i = 0; i < size; i++) {

    out->error[i] = (out->post_act[i] - target[i]) * out->post_act[i] *
                    (1 - out->post_act[i]);

    for (int j = 0; j < out->n_in; j++) {
      out->w[i * out->n_in + j] -= rate * out->error[i] * prev->post_act[j];
    }

    out->b[i] -= rate * out->error[i];
  }
}

void pup_backward(Pup *p, Pup *next, const float *input, float rate) {
  float dL_da;
  for (int i = 0; i < p->n_out; i++) {
    dL_da = 0;
    for (int j = 0; j < next->n_out; j++) {
      // sum of error * weight, chain rule
      dL_da += next->error[j] * next->w[j * (p->n_out) + i];
    }
    p->error[i] = dL_da * (p->post_act[i]) * (1 - (p->post_act[i]));
    // updating weights
    for (int j = 0; j < p->n_in; j++) {
      p->w[i * p->n_in + j] -= rate * p->error[i] * input[j];
    }
    p->b[i] -= rate * p->error[i];
  }
}

void back_prop(Stingray *s, const float *input, const float *target,
               float rate) {
  stingray_forward(s, input);
  compute_error(s, target, rate);

  for (int i = s->n_pups - 2; i >= 0; i--) {
    const float *input_to_l = (i == 0) ? input : s->pups[i - 1].post_act;
    pup_backward(&s->pups[i], &s->pups[i + 1], input_to_l, rate);
  }
}

// saving the actual file
void save(const Stingray *s, const char *path) {}

Stingray load(const char *path) {}
