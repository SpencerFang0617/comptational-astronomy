CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2
LDFLAGS = -lm

TARGET  = fmm
SRCS    = main.c quadtree.c operators.c
OBJS    = $(SRCS:.c=.o)
HEADERS = fmm.h quadtree.h operators.h

.PHONY: all clean debug

all: $(TARGET)

debug: CFLAGS = -std=c11 -Wall -Wextra -g -O0 -DDEBUG
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
