#ifndef DATASET_H
#define DATASET_H

typedef struct Dataset {
  unsigned char **images;
  int *label;
  int n;
} Dataset;

Dataset *parse(const char *dir);
void free_dataset(Dataset *set);
void visualize(Dataset *set);

#endif
