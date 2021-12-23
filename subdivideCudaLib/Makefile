# --- MACROS
# define program name
MAIN= cudaSubdivision

# define the C compiler to use
CC= nvcc

# define any compile-time flags
CFLAGS= -O3 -pg -rdc=true 

LIBNAME= -Xcompiler -fPIC -o ${MAIN}.so --shared 

# define any libraries to link into executable
LIBS= -lm

# define C source files
SRCS= subdivide.cu

# define C header files
HDRS= subdivide.h

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