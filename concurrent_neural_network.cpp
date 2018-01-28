/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2018  Daniel Darias Sánchez <dariasteam94@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "concurrent_neural_network.h"

concurrent_neural_network::concurrent_neural_network(const std::vector<std::vector<bool>>& vec_graph,
                                                     const std::vector<std::vector<double>>& vec_costs,
                                                     unsigned int inps, unsigned int outs) :
                                                     inputs(inps),
                                                     outputs(outs) {

    auto net_graph = vec_graph;
    auto net_costs = vec_costs;

    unsigned old_size = 0;
    unsigned new_size = 0;
    unsigned counter = 0;

    //  OPTIMIZE THE NET  -----------------------------------------------------

    do {
      counter++;
      old_size = net_graph.size();
      delete_unreachable_nodes(net_graph, net_costs, 3, 2);
      delete_deathend_nodes(net_graph, net_costs, 3, 2);
      new_size = net_graph.size();
    } while (old_size != new_size);

    // CALCULATE CONCURRENT NEURONS -------------------------------------------

    concurrent_steps = generate_concurrent_steps(net_graph);

    // BUILD THE NET ----------------------------------------------------------

    unsigned size = net_graph.size();
    neurons.resize(size);

    // regular neurons
    for (unsigned i = 0; i < size - outs; i++)
      neurons[i] = new neuron();

    // output neurons (can check no inputs)
    for (unsigned i = size - outs; i < size; i++)
      neurons[i] = new output_neuron();


    // Generate the net
    for (unsigned i = 0; i < size; i++) {
      // diagonal (threshold)
      neurons[i]->set_threshold(net_costs[i][i]);

      // upper triangle (propagative)
      for (unsigned j = i + 1; j < size; j++) {
        if (net_graph[i][j]) {
          axon* aux = new axon(net_costs[i][j]);
          neurons[i]->add_output(aux);
          neurons[j]->add_input(aux);
        }
      }
      // lower triangle (feedbacker)
      for (unsigned j = 0; j < i; j++) {
        if (net_graph[i][j]) {
          add_feedbacker(i, j, net_costs[i][j]);
        }
      }
    }

    // Generate the inputs axons
    input_axons.resize(inputs);
    for (unsigned i = 0; i < inputs; i++) {
      axon* aux = new axon(1);
      neurons[i]->add_input(aux);
      input_axons[i] = aux;
    }

    // Generate the outputs axons
    output_axons.resize(outputs);
    for (unsigned i = 0; i < outputs; i++) {
      axon* aux = new axon(1);
      neurons[size - outputs + i]->add_output(aux);
      output_axons[i] = aux;
    }
  }

bool concurrent_neural_network::operator()(const std::vector< double >& inputs_values,
                                           std::vector< double >& outputs_values) {

    propagate_feedback();

    // Comprobar compatibilidad de los vectores
    unsigned i_size = inputs_values.size();
    unsigned o_size = outputs;
    if (i_size != inputs)
      return false;

    // Establecer los inputs
    for (unsigned i = 0; i < i_size; i++)
      input_axons[i]->set_value(inputs_values[i]);


    std::vector<std::future<void>> promises (neurons.size());
    auto calculate_neuron = [&](unsigned i) {
      neurons[i]->calculate_value();
      neurons[i]->propagate_value();
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

void concurrent_neural_network::propagate_feedback(){
  for (auto& feedbacker : feedbackers)
    feedbacker.second->propagate_value();
}

void concurrent_neural_network::add_feedbacker(unsigned int origin_neuron, unsigned int destiny_neuron, double w){
  if (feedbackers.find(origin_neuron) == feedbackers.end())
    feedbackers[destiny_neuron] = new feedback_bus(neurons[destiny_neuron]);
  feedbackers[destiny_neuron]->add_connection(neurons[origin_neuron], w);
}


void concurrent_neural_network::delete_unreachable_nodes(std::vector<std::vector<bool>>& vec_graph,
                                                         std::vector<std::vector<double>>& vec_costs,
                                                         unsigned int inputs, unsigned int outputs) {

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


void concurrent_neural_network::delete_deathend_nodes(std::vector< std::vector< bool > >& vec_graph,
                                                      std::vector< std::vector< double > >& vec_costs,
                                                      unsigned int inputs, unsigned int outputs) {
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

std::vector< unsigned int > concurrent_neural_network::generate_visited_nodes(const std::vector< std::vector< bool > >& vec) {
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

std::vector<unsigned int> concurrent_neural_network::generate_concurrent_steps(const std::vector<std::vector<bool>>& vec) {
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
