./main : ./main.c
	gcc main.c -o main -ggdb3 -mavx -lm -lpthread