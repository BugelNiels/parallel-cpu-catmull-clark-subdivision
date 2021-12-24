# --- MACROS
# define program name
MAIN= libCudaSubdivision

# define the C compiler to use
CC= nvcc

# define any compile-time flags
CFLAGS= -O3 -pg -rdc=true -g -G

LIBNAME= -Xcompiler -fPIC -o ${MAIN}.so --shared 

# define any libraries to link into executable
LIBS= -lm

# define C source files
# TODO: use wildcards
SRCS= src/subdivide.cu src/deviceCommunication.cu src/kernelInvoker.cu src/mesh.cu src/kernels/quadRefinement.cu src/util/util.cu

# define C header files
HDRS= libheaders/subdivide.h src/util/util.cuh src/mesh.cuh src/deviceCommunication.cuh src/kernelInvoker.cuh  src/kernels/quadRefinement.cuh 

# --- TARGETS
all: ${MAIN}

#Builds the program
${MAIN}: ${SRCS} ${HDRS}
	@echo #
	@echo "-- BUILDING PROGRAM --"
	${CC} ${SRCS} ${CFLAGS} ${LIBS} ${LIBNAME} 

clean:
	@echo #
	@echo "-- CLEANING PROJECT FILES --"
	$(RM) *.o ${MAIN}