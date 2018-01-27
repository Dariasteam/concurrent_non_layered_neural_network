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

concurrent_neural_network::concurrent_neural_network(
  const std::vector< std::vector<bool>>& net_graph,
  const std::vector< std::vector< double > >& net_costs,
  const std::vector< unsigned int > concurrent_s,
  unsigned int inps, unsigned int outs):
  inputs(inps),
  outputs(outs),
  concurrent_steps (concurrent_s)
  {
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
