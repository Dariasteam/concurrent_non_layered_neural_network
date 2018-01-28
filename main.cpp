#include <iostream>
#include <vector>
#include <functional>
#include <future>
#include <stack>
#include <map>
#include <cmath>
#include <ctime>
#include <chrono>
#include <fstream>
#include <string>

#include "neuron.h"
#include "axon.h"
#include "feedback_bus.h"
#include "concurrent_neural_network.h"


template <class T>
void print_matrix (const std::vector<std::vector<T>>& graph) {
  unsigned size = graph.size();

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      std::cout << graph[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n\n";
}


std::vector<std::vector<bool>> random_graph_generator(unsigned N) {
  std::vector<std::vector<bool>> vec;
  unsigned size = N + 1;
  vec.resize(size);
  for (unsigned i = 0; i < size; i++) {
    vec[i].resize(size);
    for (unsigned j = 0; j < size; j++) {
      if (rand() % 15 < 1)
        vec[i][j] = true;
      else
        vec[i][j] = false;
    }
  }
  return vec;
}

std::vector<std::vector<double>> random_costs_generator(unsigned N) {
  std::vector<std::vector<double>> vec;
  unsigned size = N + 1;
  vec.resize(size);
  for (unsigned i = 0; i < size; i++) {
    vec[i].resize(size);
    for (unsigned j = 0; j < size; j++)
      vec[i][j] = double(-1000 + (std::rand() % 2000)) / 1000;
  }
  return vec;
}

void read_net_from_file (std::string filename, std::vector<std::vector<bool>>& graph,
                                               std::vector<std::vector<double>>& costs) {
  std::ifstream file;

  graph.resize(0);
  costs.resize(0);

  unsigned size;

  file.open(filename);
  if (file.is_open()) {
    file >> size;
    graph.resize(size);
    costs.resize(size);

    // READ GRAPH

    for (unsigned i = 0; i < size; i++) {
      graph[i].resize(size);
      costs[i].resize(size);
      for (unsigned j = 0; j < size; j++) {
        bool aux;
        file >> aux;
        graph[i][j] = aux;
      }
    }

    // READ COSTS

    for (unsigned i = 0; i < size; i++) {
      for (unsigned j = 0; j < size; j++) {
        double aux;
        file >> aux;
        costs[i][j] = aux;
      }
    }
  }
}

int main(int argc, char **argv) {
  srand(time(nullptr));
  std::vector<std::vector<bool>> vec_graph;
  std::vector<std::vector<double>> vec_costs;
/*
  vec_graph = random_graph_generator(50);
  vec_costs = random_costs_generator(50);
*/

  read_net_from_file ("testfile.dat", vec_graph, vec_costs);


  unsigned n_networks = 80;

  std::vector<concurrent_neural_network*> c_nns (n_networks);
  std::vector<std::future<void>> promises (n_networks);

  auto op_generate = [&](unsigned i) {
    c_nns[i] = new concurrent_neural_network (vec_graph, vec_costs, 3, 2);
  };

  for (unsigned i = 0; i < n_networks; i++)
    promises[i] = std::async(op_generate, i);

  for (unsigned i = 0; i < n_networks; i++)
    promises[i].get();

  std::cout << "Redes generadas" << std::endl;
  std::cout << "Net is calculated in " << c_nns[0]->c_steps()
  << " concurrent steps" << std::endl;

  std::vector<double> inputs{1, 1, 1};

  auto op_evaluate = [&](unsigned i) {
    std::vector<double> outputs;
    c_nns[i]->operator() (inputs, outputs);
    std::cout << outputs[0] << ' ' << outputs[1] << std::endl;
  };

  auto begin = std::chrono::high_resolution_clock::now();
  unsigned counter = 0;
  while (true) {
    for (unsigned i = 0; i < n_networks; i++)
      promises[i] = std::async(op_evaluate, i);

    for (unsigned i = 0; i < n_networks; i++)
      promises[i].get();

    if (counter < 10) {
      counter++;
    } else {
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      std::cout << "Fin evaluación " << time / 1000 << std::endl;
      counter = 0;
      begin = std::chrono::high_resolution_clock::now();
    }
  }
  return 0;
}
