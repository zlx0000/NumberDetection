#include "process.h"

void process_l1()
{
	int i, j;
	__m256 w10p[16][98], l0p[98], l1m[16][98];
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

void process_l2()
{
	int i, j;
	__m256 w21p[16][2], l1p[2], l2m[16][2];
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

void process_l3()
{
	int i, j;
	__m256 w32p[10][2], l2p[2], l3m[10][2];
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

void process()
{
	process_l1();
	process_l2();
	process_l3();
}