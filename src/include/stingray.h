typedef struct {
  // how many neurons each MLP reads and outputs to
  int n_in;
  int n_out;

  float *w; // # of weights --> n_out * n_in
  float *b; // # of biases --> n_out

  float *pre_act;  // z = Wx + b --> weightd sum before activation function
  float *post_act; // activation(z)

  float *error; // dL/dz for this Pup, assigned during backprop
} Pup;

typedef struct {
  Pup *pups;
  int n_pups;
} Stingray;

// construct the neural network
Stingray *stingray_build(const int *num_pups, int n);
void free_stingray(Stingray *s);

// forward prop
void pup_forward(Pup *p, const float *in);
float *stingray_forward(Stingray *s, const float *in);

// backward prop
void compute_error(Stingray *s, const float *target, float rate);

void pup_backward(Pup *p, Pup *prev, const float *input_to_l, float rate);

void back_prop(Stingray *s, const float *input, const float *target,
               float rate);

// saving
void save(const Stingray *s, const char *path);

Stingray load(const char *path);
