#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>

float *l0;
float l1[16];
float l2[16];
float l3[10];

float w10[16][784];
float w21[16][16];
float w32[10][16];

float b10[16];
float b21[16];
float b32[10];

float *y_3;
float y_2[16];
float y_1[16];

float train_sets[60000][784];
float train_sets_lables[60000][10] = {0};

float *all[13002];
float grad[13002];

float sigmoid(float x)
{
	return 1 / (1 + exp(-1 * x));
}

float sigmoid_d(float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

float ReLU(float x)
{
	if (x <= 0.0)
	{
		return 0.0;
	}
	else
	{
		return x;
	}
}

void load_train()
{
	int i, j;
	unsigned char lable, pixel;
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

__m256 w10p[16][98], l0p[98], l1m[16][98];
__m256 w21p[16][2], l1p[2], l2m[16][2];
__m256 w32p[10][2], l2p[2], l3m[10][2];

void forward_l1()
{
	int i, j;
	for (j = 0; j < 98; j++)
	{
		l0p[j] = _mm256_setr_ps(l0[j * 8], l0[j * 8 + 1], l0[j * 8 + 2], l0[j * 8 + 3], l0[j * 8 + 4], l0[j * 8 + 5], l0[j * 8 + 6], l0[j * 8 + 7]);
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 98; j++)
		{
			w10p[i][j] = _mm256_setr_ps(w10[i][j * 8], w10[i][j * 8 + 1], w10[i][j * 8 + 2], w10[i][j * 8 + 3], w10[i][j * 8 + 4], w10[i][j * 8 + 5], w10[i][j * 8 + 6], w10[i][j * 8 + 7]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 98; j++)
		{
			l1m[i][j] = _mm256_mul_ps(w10p[i][j], l0p[j]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 97; j++)
		{
			l1m[i][j + 1] = _mm256_add_ps(l1m[i][j], l1m[i][j + 1]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		l1[i] = sigmoid(l1m[i][97][0] + l1m[i][97][1] + l1m[i][97][2] + l1m[i][97][3] + l1m[i][97][4] + l1m[i][97][5] + l1m[i][97][6] + l1m[i][97][7] + b10[i]);
	}
}

void forward_l2()
{
	int i, j;
	for (j = 0; j < 2; j++)
	{
		l1p[j] = _mm256_setr_ps(l1[j * 8], l1[j * 8 + 1], l1[j * 8 + 2], l1[j * 8 + 3], l1[j * 8 + 4], l1[j * 8 + 5], l1[j * 8 + 6], l1[j * 8 + 7]);
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 2; j++)
		{
			w21p[i][j] = _mm256_setr_ps(w21[i][j * 8], w21[i][j * 8 + 1], w21[i][j * 8 + 2], w21[i][j * 8 + 3], w21[i][j * 8 + 4], w21[i][j * 8 + 5], w21[i][j * 8 + 6], w21[i][j * 8 + 7]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 2; j++)
		{
			l2m[i][j] = _mm256_mul_ps(w21p[i][j], l1p[j]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		l2m[i][1] = _mm256_add_ps(l2m[i][0], l2m[i][1]);
	}
	for (i = 0; i < 16; i++)
	{
		l2[i] = sigmoid(l2m[i][1][0] + l2m[i][1][1] + l2m[i][1][2] + l2m[i][1][3] + l2m[i][1][4] + l2m[i][1][5] + l2m[i][1][6] + l2m[i][1][7] + b21[i]);
	}
}

void forward_l3()
{
	int i, j;
	for (j = 0; j < 2; j++)
	{
		l2p[j] = _mm256_setr_ps(l2[j * 8], l2[j * 8 + 1], l2[j * 8 + 2], l2[j * 8 + 3], l2[j * 8 + 4], l2[j * 8 + 5], l2[j * 8 + 6], l2[j * 8 + 7]);
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 2; j++)
		{
			w32p[i][j] = _mm256_setr_ps(w32[i][j * 8], w32[i][j * 8 + 1], w32[i][j * 8 + 2], w32[i][j * 8 + 3], w32[i][j * 8 + 4], w32[i][j * 8 + 5], w32[i][j * 8 + 6], w32[i][j * 8 + 7]);
		}
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 2; j++)
		{
			l3m[i][j] = _mm256_mul_ps(w32p[i][j], l2p[j]);
		}
	}
	for (i = 0; i < 10; i++)
	{
		l3m[i][1] = _mm256_add_ps(l3m[i][0], l3m[i][1]);
	}
	for (i = 0; i < 10; i++)
	{
		l3[i] = sigmoid(l3m[i][1][0] + l3m[i][1][1] + l3m[i][1][2] + l3m[i][1][3] + l3m[i][1][4] + l3m[i][1][5] + l3m[i][1][6] + l3m[i][1][7] + b32[i]);
	}
}

void forward()
{
	forward_l1();
	forward_l2();
	forward_l3();
}

float db3[10], db2[16], db1[16];
__m256 db3p[10], dw32[16][10], dw32p[10][2], dl2p[2];
__m256 db2p[16], dw21[16][16], dw21p[16][2], dl1p[2];
__m256 db1p[16], dw10[784][16], dw10p[98][2];

void backprop_l3()
{
	int i, j;
	y_3 = train_sets_lables[1];
	for (i = 0; i < 10; i++)
	{
		db3[i] = l3[i] * (1 - l3[i]) * (l3[i] - y_3[i]); //Caculating the gradient.
	}
	for (i = 0; i < 10; i++)
	{
		db3p[i] = _mm256_set1_ps(db3[i]);
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 2; j++)
		{
			dw32p[i][j] = _mm256_mul_ps(l2p[j] * 2, db3p[i]);
		}
	}
	dl2p[0] = _mm256_mul_ps(dw32p[0][0], db3p[0]);
	dl2p[1] = _mm256_mul_ps(dw32p[0][1], db3p[0]); //For the first iteration, we don't have to add the result from the previous multiplication.
	for (i = 1; i < 10; i++)
	{
		for (j = 0; j < 2; j++)
		{
			dl2p[j] = _mm256_fmadd_ps(dw32p[i][j], db3p[i], dl2p[j]);
		}
	}
}

void backprop_l2()
{
	int i, j;
	_mm256_store_ps(&y_1[0], dl1p[0]);
	_mm256_store_ps(&y_1[8], dl1p[1]);
	for (i = 0; i < 16; i++)
	{
		db2[i] = l2[i] * (1 - l2[i]) * (l2[i] - y_2[i]);
	}
	for (i = 0; i < 16; i++)
	{
		db2p[i] = _mm256_set1_ps(db3[i]);
	}
	dl1p[0] = _mm256_mul_ps(dw21p[0][0], db2p[0]);
	dl1p[1] = _mm256_mul_ps(dw21p[0][1], db2p[0]);
	for (i = 1; i < 16; i++)
	{
		for (j = 0; j < 2; j++)
		{
			dw21p[i][j] = _mm256_mul_ps(l1p[j] * 2, db2p[i]);
		}
	}
}

void backprop_l1()
{
	int i, j;
	_mm256_store_ps(&y_1[0], dl1p[0]);
	_mm256_store_ps(&y_1[8], dl1p[1]);
	for (i = 0; i < 16; i++)
	{
		db2[i] = l2[i] * (1 - l2[i]) * (l2[i] - y_2[i]);
	}
	for (i = 0; i < 16; i++)
	{
		db2p[i] = _mm256_set1_ps(db3[i]);
	}
	for (i = 0; i < 98; i++)
	{
		for (j = 0; j < 2; j++)
		{
			dw10p[i][j] = _mm256_mul_ps(l0p[j] * 2, db2p[i]);
		}
	}
	dl1p[0] = _mm256_mul_ps(dw21p[0][0], db2p[0]);
	dl1p[1] = _mm256_mul_ps(dw21p[0][1], db2p[0]);
	for (i = 1; i < 16; i++)
	{
		for (j = 0; j < 2; j++)
		{
			dl1p[j] = _mm256_fmadd_ps(dw21p[i][j], db2p[i], dl1p[j]);
		}
	}
}

void backprop()
{
	backprop_l3();
	backprop_l2();
	backprop_l1();
}

void randomize()
{
	int i;
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
	forward();
	backprop();
	return 0;
}
