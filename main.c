#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <FreeImage.h>
#include "main.h"
#include "process.h"

float train_sets[60000][784];
float train_sets_lables[60000][10] = {0};

float *all[13002];
float grad[13002];

void load_train()
{
	int i, j;
	uint8_t lable, pixel;
	FILE *fp_lable_train = fopen("./train-labels.idx1-ubyte", "r");
	if (!fp_lable_train)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}
	fseek(fp_lable_train, 8, SEEK_SET);
	for (i = 0; i < 60000; i++)
	{
		lable = fgetc(fp_lable_train);
		train_sets_lables[i][(unsigned int)lable] = 1.0;
	}
	fclose(fp_lable_train);
	FILE *fp_image_train = fopen("./train-images.idx3-ubyte", "r");
	if (!fp_image_train)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}
	fseek(fp_image_train, 16, SEEK_SET);
	for (i = 0; i < 60000; i++)
	{
		for (j = 0; j < 784; j++)
		{
			pixel = fgetc(fp_image_train);
			train_sets[i][j] = ((float)((unsigned int)pixel)) / 255.0;
		}
	}
	fclose(fp_image_train);
}

void load_file(const char *filenane)
{
	int x, y, k = 0;
	FIBITMAP *img = FreeImage_Load(FIF_BMP, filenane, BMP_DEFAULT);
	for (y = 0; y < 28; y++)
	{
		for (x = 0; k < 28; x++)
		{
			l0[k] =
				k++;
		}
	}
}

void init()
{
	int i, j, k = 0;
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 784; j++)
		{
			all[k] = &(w10[i][j]);
			k++;
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 16; j++)
		{
			all[k] = &(w21[i][j]);
			k++;
		}
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 16; j++)
		{
			all[k] = &(w32[i][j]);
			k++;
		}
	}
	for (i = 0; i < 16; i++)
	{
		all[k] = &(b10[i]);
		k++;
	}
	for (i = 0; i < 16; i++)
	{
		all[k] = &(b21[i]);
		k++;
	}
	for (i = 0; i < 10; i++)
	{
		all[k] = &(b32[i]);
		k++;
	}
}

void train()
{
	load_train();
	int i, j;
	float cost = 0;
	for (i = 0; i < 60000; i++)
	{
		l0 = train_sets[i];
		process();
		for (j = 0; j < 10; j++)
		{
			cost += (l3[j] - train_sets_lables[i][j]) * (l3[j] - train_sets_lables[i][j]);
		}
	}
}

void randomize()
{
	int i, j;
	for (i = 0; i < 13002; i++)
	{
		*all[i] = (((float)rand() / (float)RAND_MAX) - 0.5);
	}
}

int main(int argc, char **argv)
{
	init();
	randomize();
	load_train();
	l0 = train_sets[0];
	process();
	return 0;
}
