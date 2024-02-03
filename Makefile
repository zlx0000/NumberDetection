debug: main.c
	gcc main.c -o main -g3 -mavx -mfma -lm -lpthread -Wall
release: main.c
	gcc main.c -o main -O3 -mavx -mfma -lm -lpthread -Wall
#-lfreeimage