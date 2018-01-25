#include <iostream>
#include <vector>
#include <functional>
#include <future>

#include <time.h>

class axon {
private:
  double value;
public:
  void set_value (double v) { value = v; }
  double get_value () const { return value; }
};

class neuron {
private:
  double value;
  std::vector<axon*> inputs;
  std::vector<axon*> outputs;
public:
  void add_input (axon* input) {
    inputs.push_back(input);
  }

  void add_output (axon* output) {
    outputs.push_back(output);
  }

  void calculate_value () {
    for (axon* aux : inputs)
      value += aux->get_value();
  }

  void propagatt_value () {
    for (axon* aux : outputs)
      aux->set_value(value);
  }

  double get_value () const { return value; }
};

class concurrent_neural_netowrk {
private:
  unsigned inputs;
  unsigned outputs;

  std::vector<axon*> input_axons;
  std::vector<axon*> output_axons;

  std::vector<neuron*> neurons;
  std::vector<unsigned> concurrent_steps;

public:

  concurrent_neural_netowrk (const std::vector<std::vector<bool>>& net,
                             const std::vector<unsigned> concurrent_s,
                             unsigned inps, unsigned outs) :
      inputs(inps),
      outputs(outs),
      concurrent_steps (concurrent_s)
    {

    unsigned size = net.size();

    neurons.resize(size);
    for (unsigned i = 0; i < size; i++)
      neurons[i] = new neuron();

    // Generate the net
    for (unsigned i = 0; i < size; i++) {
      for (unsigned j = 0; j < size; j++) {
        if (net[i][j]) {
          axon* aux = new axon();
          neurons[i]->add_output(aux);
          neurons[j]->add_input(aux);
        }
      }
    }

    // Generate the inputs axons
    input_axons.resize(inputs);
    for (unsigned i = 0; i < inputs; i++) {
      axon* aux = new axon();
      neurons[i]->add_input(aux);
      input_axons[i] = aux;
    }

    // Generate the outputs axons
    output_axons.resize(outputs);
    for (unsigned i = 0; i < outputs; i++) {
      axon* aux = new axon();
      neurons[size - outputs + i]->add_output(aux);
      output_axons[i] = aux;
    }
  }

  bool operator () (const std::vector<double>& inputs_values,
                          std::vector<double>& outputs_values) {

    // Comprobar compatibilidad de los vectores
    unsigned i_size = inputs_values.size();
    unsigned o_size = outputs;
    if (i_size != inputs)
      return false;

    // Establecer los inputs
    for (unsigned i = 0; i < i_size; i++)
      input_axons[i]->set_value(inputs_values[i]);


    std::vector<std::future<void>> promises (neurons.size());
    auto calculate_neuron = [&](unsigned i ) {
      neurons[i]->calculate_value();
      neurons[i]->propagatt_value();
    };

    // Realizar el cálculo concurrente
    unsigned last_neuron = 0;
    for (unsigned concurrent_group : concurrent_steps) {
      for (unsigned i = last_neuron; i <= concurrent_group; i++)
        promises[i] = std::async(calculate_neuron, i);

      for (unsigned i = last_neuron; i < concurrent_group; i++)
        promises[i].get();

      last_neuron = concurrent_group + 1;
    }

    // Recoger los outputs
    outputs_values.resize(o_size);
    for (unsigned i = 0; i < o_size; i++)
      outputs_values[i] = output_axons[i]->get_value();

    return true;
  }
};


std::vector<std::vector<bool>> random_graph_generator() {
  std::vector<std::vector<bool>> vec;
  unsigned size = rand() % 10000 + 1;
  vec.resize(size);
  for (unsigned i = 0; i < size; i++) {
    vec[i].resize(size);
    for (unsigned j = i; j < size; j++) {
      if (rand() % 5 < 1)
        vec[i][j] = true;
      else
        vec[i][j] = false;
    }
  }
  return vec;
}

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

int main(int argc, char **argv) {
  srand(time(nullptr));
  auto vec_graph = random_graph_generator();

  /*
  vec_graph = {
    {0, 0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0, 1},
    {0, 0, 0, 0, 1, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  */

  auto vec_solve = generate_concurrent_steps(vec_graph);

  unsigned size = vec_graph.size();

  /*
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      std::cout << vec_graph[i][j] << " ";
    }
    std::cout << "\n";
  }
  */

  std::cout << "\n\n";

  size = vec_solve.size();
  for (unsigned i = 0; i < size; i++)
    std::cout << vec_solve[i] << " ";

  concurrent_neural_netowrk nn (vec_graph, vec_solve, 3, 2);

  std::vector<double> outputs;
  std::vector<double> inputs{1, 1, 1};

  std::cout << "\n\n" << vec_graph.size() << std::endl;

  while (1)
    nn (inputs, outputs);

  if (nn (inputs, outputs)) {
    std::cout << "\nSuccess\n";
    for (auto& value : outputs)
      std::cout << value << " ";
  } else {
    std::cout << "\nError\n";
  }

  return 0;
}
