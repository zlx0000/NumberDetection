debug: main.c
	gcc main.c -o main -g -mavx -lm -lpthread -lfreeimage
release: main.c
	gcc main.c -o main -g -O3 -mavx -lm -lpthread -lfreeimage
