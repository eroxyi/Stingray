#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float **input;
  int *labels;
  int n;
} Dataset;

Dataset *parse(const char *path, const char *path2) {
  Dataset *set = (Dataset *)(malloc(sizeof(Dataset)));
  FILE *file_ptr;

  if ((file_ptr = fopen(path, "rb")) == NULL) {
    printf("File reading failed!\n");
    exit(1);
  }

  fseek(file_ptr, 4, SEEK_SET);
  fread(&(set->n), 4, 1, file_ptr);
  fseek(file_ptr, 16, SEEK_SET);

  // loop through and populate input

  // loop through and populate label

  fclose(file_ptr);
}
