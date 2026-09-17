#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_IMAGE 1000
#define PIXELS 784
#define MAX_FILES 10

typedef struct Dataset {
  unsigned char **images;
  int *label;
  int n;
} Dataset;

Dataset *parse(const char *dir) {
  Dataset *temp = (malloc(sizeof(Dataset)));

  temp->images = (malloc(NUM_IMAGE * sizeof(char *) * MAX_FILES));
  temp->label = (malloc(sizeof(int) * NUM_IMAGE * MAX_FILES));
  temp->n = NUM_IMAGE;

  DIR *dirFile = opendir(dir);
  struct dirent *hFile;

  int counter = 0;
  while ((hFile = readdir(dirFile)) != NULL) {
    if (!strcmp(hFile->d_name, "."))
      continue;
    if (!strcmp(hFile->d_name, ".."))
      continue;

    char fullPath[250];
    strcpy(fullPath, dir);
    strcat(fullPath, hFile->d_name);

    FILE *fptr = fopen(fullPath, "rb");

    for (int i = 0; i < NUM_IMAGE; i++) {
      temp->images[counter] = malloc(sizeof(char) * PIXELS);
      fread(temp->images[counter], sizeof(char) * PIXELS, 1, fptr);
      temp->label[counter] = hFile->d_name[4];
      counter++;
    }

    fclose(fptr);
  }

  return temp;
}

void visualize(Dataset *s) {
  char symbols[] = ".-+*#";
  printf("This should be outputting the number %d", *(s->label + 1) - '0');

  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      if (s->images[1][28 * i + j] > 128) {
        printf("#");
      } else {
        printf(" ");
      }
    }
    printf("\n");
  }
}

int main() {
  Dataset *set = parse("../data/");
  visualize(set);
}
