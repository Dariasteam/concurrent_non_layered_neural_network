CC=g++
CXXFLAGS=-g -std=c++11 -fPIC -pthread -O3

OBJS = neuron.o feedback_bus.o concurrent_neural_network.o main.o

default: ${OBJS}
	$(CC) $(CXXFLAGS) -o concurrent_graph ${OBJS}

clean:
	rm -rf *.o concurrent_graph
