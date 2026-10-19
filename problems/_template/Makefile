CC     ?= gcc
CFLAGS ?= -O2 -std=c11

prog: src/main.c src/solve.c src/solve.h
	$(CC) $(CFLAGS) src/main.c src/solve.c -o prog -lm

clean:
	rm -f prog

.PHONY: clean
