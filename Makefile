ROOT_DIR= $(shell pwd)
TARGETS= toolkits/bc toolkits/bfs toolkits/cc toolkits/pagerank toolkits/sssp toolkits/hyper_sssp toolkits/hyper_pagerank toolkits/hyper_mis toolkits/hyper_bfs toolkits/hyper_cc toolkits/hyper_bpath toolkits/hyper_bc toolkits/hyper_kcore
MACROS= 
# MACROS= -D PRINT_DEBUG_MESSAGES

MPICXX= mpicxx
CXXFLAGS= -O3 -Wall -std=c++11 -g -fopenmp -march=native -I$(ROOT_DIR) $(MACROS)
SYSLIBS= -lnuma
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

toolkits/%: toolkits/%.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean: 
	rm -f $(TARGETS)

