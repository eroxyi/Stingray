typedef struct {
  // input size and output size (number of neurons)
  int n_in;
  int n_out;

  float *w; // # of weights --> n_out * n_in
  float *b; // # of biases --> n_out

  float *pre_act;  // z = Wx + b --> weightd sum before activation function
  float *post_act; // activation(z)

  float *d_pre_act; // dL/dz for this Pup, assigned during backprop
} Pup;

typedef struct {
  Pup *pups;
  int n_pups;
} Stingray;
