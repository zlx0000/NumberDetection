ALL: main

process.o: process.c process.h
	gcc -c process.c -o process.o -g -mavx

gradient.h: gradient.c gradient.h
	gcc -c gradient.c -o gradient.o -g -mavx

sigmoid.o: sigmoid.c sigmoid.h
	gcc -c sigmoid.c -o sigmoid.o -g -lm

main: main.c process.o gradient.o sigmoid.o
	gcc main.c process.o gradient.o sigmoid.o -o main -g -lpthread -lfreeimage 

clean:
	rm main
	rm *.o