# --- MACROS
# define program name
MAIN= subdivide

# define the C compiler to use
CC= nvcc

# define any compile-time flags
CFLAGS= -O3 -pg -rdc=true -g -G

# define any libraries to link into executable
LIBS= -lm

# define C source files
# TODO: use wildcards
SRCS= ${wildcard src/*.cu} ${wildcard src/**/*.cu}

# define C header files
HDRS= libheaders/subdivide.h ${wildcard src/*.cuh} ${wildcard src/**/*.cuh}

# --- TARGETS
all: ${MAIN}

#Builds the program
${MAIN}: ${SRCS} ${HDRS}
	@echo #
	@echo "-- BUILDING PROGRAM --"
	${CC} ${SRCS} ${CFLAGS} ${LIBS} -o ${MAIN}

clean:
	@echo #
	@echo "-- CLEANING PROJECT FILES --"
	$(RM) *.o ${MAIN}