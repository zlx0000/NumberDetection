#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <FreeImage.h>

float l0[784];
float l1[16];
float l2[16];
float l3[10];

float p10[16][784];
float p21[16][16];
float p32[10][16];

float b10[16];
float b21[16];
float b32[10];

float *(loss)[13002];

float sigmoid(float x)
{
	return 1 / (1 + exp(-1 * x));
}

void init()
{
	int i, j, k = 0;
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 784; j++)
		{
			loss[k] = &(p10[i][j]);
			k++;
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 16; j++)
		{
			loss[k] = &(p21[i][j]);
			k++;
		}
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 16; j++)
		{
			loss[k] = &(p32[i][j]);
			k++;
		}
	}
	for (i = 0; i < 16; i++)
	{
		loss[k] = &(b10[i]);
		k++;
	}
	for (i = 0; i < 16; i++)
	{
		loss[k] = &(b21[i]);
		k++;
	}
	for (i = 0; i < 10; i++)
	{
		loss[k] = &(b32[i]);
		k++;
	}
}

void process_l1()
{
	int i, j;
	__m256 p10p[16][98], l0p[98], l1m[16][98];
	for (j = 0; j < 98; j++)
	{
		l0p[j] = _mm256_setr_ps(l0[j * 8], l0[j * 8 + 1], l0[j * 8 + 2], l0[j * 8 + 3], l0[j * 8 + 4], l0[j * 8 + 5], l0[j * 8 + 6], l0[j * 8 + 7]);
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 98; j++)
		{
			p10p[i][j] = _mm256_setr_ps(p10[i][j * 8], p10[i][j * 8 + 1], p10[i][j * 8 + 2], p10[i][j * 8 + 3], p10[i][j * 8 + 4], p10[i][j * 8 + 5], p10[i][j * 8 + 6], p10[i][j * 8 + 7]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 98; j++)
		{
			l1m[i][j] = _mm256_mul_ps(p10p[i][j], l0p[j]);
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

void process_l2()
{
	int i, j;
	__m256 p21p[16][2], l1p[2], l2m[16][2];
	for (j = 0; j < 2; j++)
	{
		l1p[j] = _mm256_setr_ps(l1[j * 8], l1[j * 8 + 1], l1[j * 8 + 2], l1[j * 8 + 3], l1[j * 8 + 4], l1[j * 8 + 5], l1[j * 8 + 6], l1[j * 8 + 7]);
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 2; j++)
		{
			p21p[i][j] = _mm256_setr_ps(p21[i][j * 8], p21[i][j * 8 + 1], p21[i][j * 8 + 2], p21[i][j * 8 + 3], p21[i][j * 8 + 4], p21[i][j * 8 + 5], p21[i][j * 8 + 6], p21[i][j * 8 + 7]);
		}
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 2; j++)
		{
			l2m[i][j] = _mm256_mul_ps(p21p[i][j], l1p[j]);
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

void process_l3()
{
	int i, j;
	__m256 p32p[10][2], l2p[2], l3m[10][2];
	for (j = 0; j < 2; j++)
	{
		l2p[j] = _mm256_setr_ps(l2[j * 8], l2[j * 8 + 1], l2[j * 8 + 2], l2[j * 8 + 3], l2[j * 8 + 4], l2[j * 8 + 5], l2[j * 8 + 6], l2[j * 8 + 7]);
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 2; j++)
		{
			p32p[i][j] = _mm256_setr_ps(p32[i][j * 8], p32[i][j * 8 + 1], p32[i][j * 8 + 2], p32[i][j * 8 + 3], p32[i][j * 8 + 4], p32[i][j * 8 + 5], p32[i][j * 8 + 6], p32[i][j * 8 + 7]);
		}
	}
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 2; j++)
		{
			l3m[i][j] = _mm256_mul_ps(p32p[i][j], l2p[j]);
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

void randomize()
{
	int i, j;
	for (i = 0; i < 784; i++)
	{
		l0[i] = ((float)rand() / (float)RAND_MAX);
	}
	for (i = 0; i < 16; i++)
	{
		b10[i] = ((float)rand() / (float)RAND_MAX);
		for (j = 0; j < 784; j++)
		{
			p10[i][j] = ((float)rand() / (float)RAND_MAX);
		}
	}
	for (i = 0; i < 16; i++)
	{
		b21[i] = ((float)rand() / (float)RAND_MAX);
		for (j = 0; j < 16; j++)
		{
			p21[i][j] = ((float)rand() / (float)RAND_MAX);
		}
	}
	for (i = 0; i < 10; i++)
	{
		b32[i] = ((float)rand() / (float)RAND_MAX);
		for (j = 0; j < 16; j++)
		{
			p32[i][j] = ((float)rand() / (float)RAND_MAX);
		}
	}
}

void process()
{
	process_l1();
	process_l2();
	process_l3();
}

int main(int argc, char **argv)
{
	init();
	randomize();
	process();
	return 0;
}
