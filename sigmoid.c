#include "sigmoid.h"

float sigmoid(float x)
{
	return 1 / (1 + exp(-1 * x));
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