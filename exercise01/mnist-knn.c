#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TRAIN_IMAGES_FILE "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels-idx1-ubyte"

#define TEST_IMAGES_FILE "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILE "t10k-labels-idx1-ubyte"

#define WIDTH 28
#define HEIGHT 28
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define K 10
#define NUM_CLASSIFICATIONS 10000

struct dist {
  float dist;
  unsigned char label;
  int i;
};

unsigned char train_img[NUM_TRAIN_IMAGES][WIDTH * HEIGHT];
unsigned char train_label[NUM_TRAIN_IMAGES];

unsigned char test_img[NUM_TEST_IMAGES][WIDTH * HEIGHT];
unsigned char test_label[NUM_TEST_IMAGES];

struct dist dists[NUM_TRAIN_IMAGES];

int dist_cmp_func(const void *a, const void *b)
{
  struct dist *da = (struct dist *)a;
  struct dist *db = (struct dist *)b;
  return da->dist - db->dist;
}

float euclid_dist(int a, int b)
{
  unsigned int sum = 0.0;

  int i = 0;
  for(i = 0; i < WIDTH * HEIGHT; i++)
  {
    int factor = train_img[a][i] - test_img[b][i];
    sum += factor * factor;
  }

  return sqrt(sum);
}

void read_files()
{
  int img_fd, label_fd;

  img_fd = open(TRAIN_IMAGES_FILE, O_RDONLY);
  label_fd = open(TRAIN_LABELS_FILE, O_RDONLY);

  if(img_fd < 0 || label_fd < 0)
  {
    printf("Cannot open training files\n");
    exit(1);
  }

  // ignore headers
  read(img_fd, &train_img, 4 * sizeof(int));
  read(label_fd, &train_img, 2 * sizeof(int));

  int i;
  for(i = 0; i < NUM_TRAIN_IMAGES; i++)
  {
    read(img_fd, &train_img[i], WIDTH * HEIGHT);
    read(label_fd, &train_label[i], 1);
  }

  close(img_fd);
  close(label_fd);

  img_fd = open(TEST_IMAGES_FILE, O_RDONLY);
  label_fd = open(TEST_LABELS_FILE, O_RDONLY);

  if(img_fd < 0 || label_fd < 0)
  {
    printf("cannot open test files");
    exit(1);
  }

  read(img_fd, &test_img, 4 * sizeof(int));
  read(label_fd, &test_img, 2 * sizeof(int));

  for(i = 0; i < NUM_TEST_IMAGES; i++)
  {
    read(img_fd, &test_img[i], WIDTH * HEIGHT);
    read(label_fd, &test_label[i], 1);
  }

  close(img_fd);
  close(label_fd);
}

void write_images(int ref)
{
  int fd = open("img/ref.pgm", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  char header[1024];

  snprintf(header, 1024, "P5\n%d %d 255\n", WIDTH, HEIGHT);

  write(fd, header, strlen(header));
  write(fd, test_img[ref], WIDTH * HEIGHT);

  close(fd);

  int k;
  for(k = 0; k < K; k++)
  { 
    char filename[1024];
    snprintf(filename, 1024, "img/nearest_%d.pgm", k);
    fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    snprintf(header, 1024, "P5\n%d %d 255\n", WIDTH, HEIGHT);

    write(fd, header, strlen(header));
    write(fd, train_img[dists[k].i], WIDTH * HEIGHT);

    close(fd);
  }
}

int main()
{
  read_files();

  int freqs[10];
  int num_correct = 0;

  int ref;
  for(ref = 0; ref < NUM_CLASSIFICATIONS; ref++)
  {
    int i;
    for(i = 0; i < NUM_TRAIN_IMAGES; i++)
    {
      dists[i].dist = euclid_dist(i, ref);
      dists[i].label = train_label[i];
      dists[i].i = i;
    }

    qsort(dists, NUM_TRAIN_IMAGES, sizeof(struct dist), dist_cmp_func);

    for(i = 0; i < 10; i++)
      freqs[i] = 0;

    for(i = 0; i < K; i++)
      freqs[dists[i].label]++;

    int max = 0;
    int max_i = 0;

    for(i = 0; i < 10; i++)
    {
      if(freqs[i] >= max)
      {
        max = freqs[i];
        max_i = i;
      }
    }

    printf("Guessed label: %d (%.2f%% of %d nearest). Actual label is %d%s\n", max_i, (float)max * 100.0 / K, K, test_label[ref], max_i == test_label[ref] ? ": correct": "");
    if(max_i == test_label[ref])
      num_correct++;

    // write_images(ref); // for debugging
  }

  printf("Accuracy: %.2f\n", (float)num_correct / NUM_CLASSIFICATIONS);

  return 0;
}
