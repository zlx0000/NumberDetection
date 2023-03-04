debug:
	gcc main.c -o main -g -mavx -lm -lpthread
release:
	gcc main.c -o main -g -O3 -mavx -lm -lpthread