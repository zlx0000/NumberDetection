debug:
	gcc main.c -o main -ggdb3 -mavx -lm -lpthread
release:
	gcc main.c -o main -ggdb3 -mavx -lm -lpthread -O3