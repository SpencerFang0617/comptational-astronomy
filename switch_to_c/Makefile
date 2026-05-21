CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -fopenmp -I.
LDFLAGS = -fopenmp -lm

TARGET  = fmm
SRCS    = main.c quadtree.c \
          caculate/binomial.c \
          caculate/P2M.c caculate/M2M.c caculate/M2L.c \
          caculate/L2L.c caculate/L2P.c caculate/P2P.c
OBJS    = $(SRCS:.c=.o)
HEADERS = fmm.h quadtree.h operators.h

.PHONY: all clean debug

all: $(TARGET)

debug: CFLAGS = -std=c11 -Wall -Wextra -g -O0 -DDEBUG -fopenmp -I.
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

ifdef OS
   RM = del /Q
   FixPath = $(subst /,\,$1)
else
   ifeq ($(shell uname), Linux)
      RM = rm -f
      FixPath = $1
   endif
endif

# 使用範例：
clean:
	$(RM) $(OBJS) $(TARGET)
