# nim-cpu-bridge shim build
# Produces: libcuda.so.1 and libcudart.so.12 (symlinks to libcuda_shim.so)

CC      = gcc
CFLAGS  = -shared -fPIC -O2 -Wall -Wextra -std=c11
LDFLAGS = -lpthread -ldl

TARGET  = libcuda_shim.so
SRCS    = cuda_shim.c

.PHONY: all clean install

all: $(TARGET) symlinks

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

symlinks: $(TARGET)
	@ln -sf $(TARGET) libcuda.so.1
	@ln -sf $(TARGET) libcuda.so
	@ln -sf $(TARGET) libcudart.so.12
	@ln -sf $(TARGET) libcudart.so
	@ln -sf $(TARGET) libnvidia-ml.so.1
	@ln -sf $(TARGET) libnvidia-ml.so
	@ln -sf $(TARGET) libcublas.so.12
	@ln -sf $(TARGET) libcublasLt.so.12
	@ln -sf $(TARGET) libnccl.so.2
	@echo "Shim symlinks created."

install: all
	install -d $(DESTDIR)/usr/lib/nim-cpu-bridge
	install -m 755 $(TARGET) $(DESTDIR)/usr/lib/nim-cpu-bridge/
	cd $(DESTDIR)/usr/lib/nim-cpu-bridge && \
		ln -sf $(TARGET) libcuda.so.1 && \
		ln -sf $(TARGET) libcuda.so && \
		ln -sf $(TARGET) libcudart.so.12 && \
		ln -sf $(TARGET) libcudart.so && \
		ln -sf $(TARGET) libnvidia-ml.so.1 && \
		ln -sf $(TARGET) libnvidia-ml.so && \
		ln -sf $(TARGET) libcublas.so.12 && \
		ln -sf $(TARGET) libcublasLt.so.12 && \
		ln -sf $(TARGET) libnccl.so.2

clean:
	rm -f $(TARGET) *.so *.so.*
