#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <inttypes.h>

#define BLOCKSIZE 32
#define NUM_BLOCKS 1875

__global__ void computeDistances(float *testimage, float *trainimages, float *dist, unsigned int width, unsigned int height) {
	unsigned int image_id = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	for (unsigned int i = 0; i < width * height; i++) {
		float factor = testimage[i] - trainimages[image_id * width * height + i];
		sum += factor * factor;
	}
	dist[image_id] = sqrt(sum);
}

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

float train_img[NUM_TRAIN_IMAGES * WIDTH * HEIGHT];
uint8_t train_label[NUM_TRAIN_IMAGES];

float test_img[NUM_TEST_IMAGES * WIDTH * HEIGHT];
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
  float sum = 0.0f;

  int i = 0;
  for(i = 0; i < WIDTH * HEIGHT; i++)
  {
    int factor = train_img[a * WIDTH * HEIGHT + i] - test_img[b * WIDTH * HEIGHT + i];
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

  int i, j;
  for(i = 0; i < NUM_TRAIN_IMAGES; i++)
  {
	  uint8_t val = 0;
	  for (j = 0; j < WIDTH * HEIGHT; j++) {
		  read(img_fd, &val, 1);
		  train_img[i*WIDTH*HEIGHT + j] = (float) val;
	  }

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
	  uint8_t val = 0;
	  for (j = 0; j < WIDTH * HEIGHT; j++) {
		  read(img_fd, &val, 1);
		  test_img[i*WIDTH*HEIGHT + j] = (float) val;
	  }
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
  uint8_t tmp_test_img[NUM_TEST_IMAGES * WIDTH * HEIGHT];
  uint8_t tmp_train_img[NUM_TRAIN_IMAGES * WIDTH * HEIGHT];
  for (unsigned int i = 0; i < NUM_TRAIN_IMAGES * WIDTH * HEIGHT; i++) {
  	tmp_train_img[i] = (uint8_t) train_img[i];
  }
  for (unsigned int i = 0; i < NUM_TEST_IMAGES * WIDTH * HEIGHT; i++) {
  	tmp_test_img[i] = (uint8_t) test_img[i];
  }
  write(fd, header, strlen(header));
  write(fd, tmp_test_img, NUM_TEST_IMAGES * WIDTH * HEIGHT);

  close(fd);

  int k;
  for(k = 0; k < K; k++)
  { 
    char filename[1024];
    snprintf(filename, 1024, "img/nearest_%d.pgm", k);
    fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    snprintf(header, 1024, "P5\n%d %d 255\n", WIDTH, HEIGHT);

    write(fd, header, strlen(header));
    write(fd, &tmp_train_img[dists[k].i * WIDTH * HEIGHT], WIDTH * HEIGHT);

    close(fd);
  }
}

int main(int argc, char **argv)
{
  printf("Reading files...\n");
  read_files();
  printf("Files read.\n");
  int parallel = 0;
  if (argc > 1 && argv[1][0] == 'p') {
	  printf("Selected parallel computation.\n");
	  parallel = 1;
  }
  cudaSetDevice(1);
  int freqs[10];
  int num_correct = 0;

  int ref;

  float *d_dists;
  float *d_images;
  float *d_testimage;

  cudaMalloc(&d_images, NUM_TRAIN_IMAGES*WIDTH*HEIGHT*sizeof(float));
  cudaMalloc(&d_dists, NUM_TRAIN_IMAGES*sizeof(float));
  cudaMalloc(&d_testimage, WIDTH*HEIGHT*sizeof(float));

  cudaMemcpy(d_images, train_img, NUM_TRAIN_IMAGES*WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice);

  printf("Starting classifications...\n");
  clock_t begin = clock();
  for(ref = 0; ref < NUM_CLASSIFICATIONS; ref++)
  {
    int i;
    if (!parallel) {
    	for(i = 0; i < NUM_TRAIN_IMAGES; i++)
    	{
    		dists[i].dist = euclid_dist(i, ref);
    		dists[i].label = train_label[i];
    		dists[i].i = i;
    	}
    } else {
    	cudaMemcpy(d_testimage, &test_img[ref*WIDTH*HEIGHT], WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice);
    	computeDistances<<<NUM_BLOCKS, BLOCKSIZE>>>(d_testimage, d_images, d_dists, WIDTH, HEIGHT);
    	float tmp_dists[NUM_TRAIN_IMAGES];
    	cudaMemcpy(tmp_dists, d_dists, NUM_TRAIN_IMAGES*sizeof(float), cudaMemcpyDeviceToHost);

    	for(i = 0; i < NUM_TRAIN_IMAGES; i++)
    	{
    		dists[i].dist = tmp_dists[i];
    		dists[i].label = train_label[i];
    		dists[i].i = i;
    	}
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

    //printf("Guessed label: %d (%.2f%% of %d nearest). Actual label is %d%s\n", max_i, (float)max * 100.0 / K, K, test_label[ref], max_i == test_label[ref] ? ": correct": "");
    if(max_i == test_label[ref])
      num_correct++;

    // write_images(ref); // for debugging
  }
  clock_t end = clock();
  printf("Classification finished, CPU-time: %f\n", (double) (end - begin) / CLOCKS_PER_SEC);
  cudaFree(d_images);
  cudaFree(d_dists);
  cudaFree(d_testimage);
  printf("Accuracy: %.2f\n", (float)num_correct / NUM_CLASSIFICATIONS);

  return 0;
}
