CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lm
WINCC = x86_64-w64-mingw32-gcc
ODIN = $(shell which odin 2>/dev/null || echo $$HOME/odin_bin/odin)

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

run_odin: src/*.odin
	$(ODIN) build src -out:run_odin -o:speed

win64: gemma4.c win.c win.h
	$(WINCC) $(CFLAGS) -static -D_WIN32 gemma4.c win.c -o run.exe $(LDFLAGS) -lshell32

.PHONY: clean win64 all
all: run run_odin

clean:
	rm -f run run.exe run_odin
