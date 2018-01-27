#include <iostream>
#include <vector>
#include <functional>
#include <future>
#include <stack>
#include <map>
#include <cmath>

#include <time.h>

#include "neuron.h"
#include "axon.h"
#include "feedback_bus.h"
#include "concurrent_neural_network.h"

/**
* @brief Generates a vector containing the number of predecesors of each node
* This function is used by #generate_concurrent_steps to which neurons can
* be calculated concurrently.
*
* @param vec p_vec:...
* @return std::vector< unsigned int >
*/
std::vector<unsigned> generate_visited_nodes (const std::vector<std::vector<bool>>& vec) {
  unsigned size = vec.size();
  std::vector<unsigned> visited_nodes (size);

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = i + 1; j < size; j++) {
      if (vec[i][j])
        visited_nodes[j]++;
    }
  }
  return visited_nodes;
}


/**
* @brief delete the row and col indicated by node
* the matrix is divided in 4 cuadrants
* [a b]
* [c b]
*
*
* @param original p_original:...
* @param node p_node:...
*/

template <class T>
void delete_row_col (std::vector<std::vector<T>>& original, unsigned node) {
  unsigned size = original.size();
  std::vector<std::vector<T>> aux (size - 1);
  for (auto& row : aux)
    row.resize(size - 1);

  // First quadrant (a)
  for (unsigned i = 0; i < node; i++) {
    for (unsigned j = 0; j < node; j++) {
      aux[i][j] = original[i][j];
    }
  }

  // Second quadrant (b)
  for (unsigned i = 0; i < node; i++) {
    for (unsigned j = node + 1; j < size; j++) {
      aux[i][j - 1] = original[i][j];
    }
  }

  // Third quadrant (c)
  for (unsigned i = node + 1; i < size; i++) {
    for (unsigned j = 0; j < node; j++) {
      aux[i - 1][j] = original[i][j];
    }
  }

  // Fourth quadrant (d)
  for (unsigned i = node + 1; i < size; i++) {
    for (unsigned j = node + 1; j < size; j++) {
      aux[i - 1][j - 1] = original[i][j];
    }
  }

  original = aux;
}


/**
* @brief Find deathend nodes to delete them and search in cascade new posible
* deathend nodes. Inputs and outputs neuron won't be affected
*
* @param vec p_vec: newral network graph
* @param inputs p_inputs: number of input neurons
* @param outputs p_outputs: number of output neurons
*/
void delete_deathend_nodes (std::vector<std::vector<bool>>& vec_graph,
                            std::vector<std::vector<double>>& vec_costs,
                            unsigned inputs, unsigned outputs) {
  unsigned size = vec_graph.size();

  std::stack<unsigned> predecesors;
  for (unsigned i = size - outputs - 1; i >= inputs; i--)
    predecesors.push(i);

  unsigned i;
  while (predecesors.size() != 0) {
    // find next valid candidate in the stack
    do {
      if (predecesors.size() == 0)
        return;
      i = predecesors.top();
      predecesors.pop();
    } while (i < inputs || i > size - outputs - 1);

    bool empty_row = true;
    for (unsigned j = i + 1; j < size; j++) {
      if (vec_graph[i][j]) {
        empty_row = false;
        j = size;
      }
    }
    // delete deathend node and find all predecesors
    if (empty_row) {
      predecesors.push(i);
      for (unsigned k = 0; k < size; k++) {
        if (vec_graph[k][i]) {
          predecesors.push(k > i ? k - 1 : k);
          vec_graph[k][i] = 0;
        }
      }
      delete_row_col<bool> (vec_graph, i);
      delete_row_col<double> (vec_costs, i);
      size--;
    }
  }
}

/**
* @brief Delete neurons with no predecesors (excluding inputs and outputs)
*
* @param vec_graph p_vec_graph:...
* @param vec_costs p_vec_costs:...
* @param inputs p_inputs:...
* @param outputs p_outputs:...
*/
void delete_unreachable_nodes (std::vector<std::vector<bool>>& vec_graph,
                                 std::vector<std::vector<double>>& vec_costs,
                                unsigned inputs, unsigned outputs) {

  unsigned size = vec_graph.size();

  unsigned j;

  std::stack<unsigned> sucesors;
  for (unsigned i = size - outputs - 1; i >= inputs; i--)
    sucesors.push(i);

  std::vector<bool> has_predecesor (size);

  while (sucesors.size() != 0) {
    // find next valid candidate in the stack
    do {
      if (sucesors.size() == 0)
        return;
      j = sucesors.top();
      sucesors.pop();
    } while (j < inputs || j > size - outputs - 1);

    bool empty_column = true;
    for (int i = j - 1; i >= 0; i--) {
      if (vec_graph[i][j]) {
        empty_column = false;
        i = 0;
      }
    }

    // delete unreachable node and find all sucessors
    if (empty_column) {
      sucesors.push(j);
      for (unsigned k = 0; k < size; k++) {
        if (vec_graph[j][k]) {
          sucesors.push(j > k ? j - 1 : j);
          vec_graph[j][k] = 0;
        }
      }
      delete_row_col<bool> (vec_graph, j);
      delete_row_col<double> (vec_costs, j);
      size--;
    }
  }
}

/**
* @brief Extracts the hidden layers of the net and creates a vector of groups
* of neurons that can be safely calculated concurrently.
*
* @param vec p_vec: Cost matriz of the net
* @return std::vector< unsigned int > groups of neurons conccurent-safe
*/
std::vector<unsigned> generate_concurrent_steps (const std::vector<std::vector<bool>>& vec) {
  unsigned size = vec.size();
  std::vector<unsigned> visited_nodes = generate_visited_nodes(vec);

  std::vector<unsigned> solve;

  auto aux_visited = visited_nodes;

  unsigned last_node = 0;

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = i + 1; j < size; j++) {
      if (vec[i][j]) {
        if (visited_nodes[i] != 0) {
          visited_nodes = aux_visited;
          solve.push_back(last_node);
        }
        aux_visited[j]--;
        last_node = i;
      }
    }
  }
  solve.push_back(last_node);
  solve.push_back(size - 1);
  return solve;
}


void print_graph_matrix (const std::vector<std::vector<bool>>& graph) {
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


int main(int argc, char **argv) {
  srand(time(nullptr));
  auto vec_graph = random_graph_generator(7000);
  auto vec_costs = random_costs_generator(7000);

/*
  vec_graph = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0},
  };
*/
  // print_graph_matrix(vec_graph);

  unsigned first_size = vec_graph.size();;
  unsigned old_size = 0;
  unsigned new_size = 0;

  unsigned counter = 0;

  do {
    counter++;
    old_size = vec_graph.size();
//    std::cout << "Unreachable" << std::endl;
    delete_unreachable_nodes(vec_graph, vec_costs, 3, 2);
//    print_graph_matrix(vec_graph);

//    std::cout << "Deathend" << std::endl;
    delete_deathend_nodes(vec_graph, vec_costs, 3, 2);
//    print_graph_matrix(vec_graph);

    new_size = vec_graph.size();
  } while (old_size != new_size);

  std::cout << std::flush;

  std::cout << "Improved from " << first_size << " neurons to "
  << vec_graph.size() << " " << " in " << counter << " steps\n";

  auto vec_solve = generate_concurrent_steps(vec_graph);

  std::cout << "\n\n";
/*
  unsigned size = vec_solve.size();
  for (unsigned i = 0; i < size; i++)
    std::cout << vec_solve[i] << " ";
*/

  std::vector<double> inputs{1, 1, 1};

  std::cout << "Net is calculated in " << vec_solve.size()
            << " concurrent steps" << std::endl;

  unsigned n_networks = 1;

  std::vector<concurrent_neural_network*> c_nns (n_networks);
  std::vector<std::future<void>> promises (n_networks);

  auto op_generate = [&](unsigned i) {
    c_nns[i] = new concurrent_neural_network (vec_graph, vec_costs, vec_solve, 3, 2);
  };

  for (unsigned i = 0; i < n_networks; i++)
    promises[i] = std::async(op_generate, i);

  for (unsigned i = 0; i < n_networks; i++)
    promises[i].get();

  std::cout << "Redes generadas" << std::endl;

  auto op_evaluate = [&](unsigned i) {
    std::vector<double> outputs;
    c_nns[i]->operator() (inputs, outputs);
  };

  while (true) {
    for (unsigned i = 0; i < n_networks; i++)
      promises[i] = std::async(op_evaluate, i);

    for (unsigned i = 0; i < n_networks; i++)
      promises[i].get();

    std::cout << "Fin evaluación" << std::endl;
  }

  return 0;
}
