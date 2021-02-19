include Makefile.inc

all: color

test : test.cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/test test.cu $(SOURCE) $(ARCH) $(INC)

color : color.cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/color color.cu $(SOURCE) $(ARCH) $(INC)

color_transform : color_transform.cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/color_transform color_transform.cu $(SOURCE) $(ARCH) $(INC)

