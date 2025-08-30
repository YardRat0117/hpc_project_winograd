CC = nvcc
CFLAGS = -O3 -std=c++17
TARGET = winograd
LDLIBS = -lcublas
SOURCES = main.cu naive_conv.cu winograd_conv.cu

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
