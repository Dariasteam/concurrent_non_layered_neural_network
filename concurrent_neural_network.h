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

#ifndef CONCURRENT_NEURAL_NETWORK_H
#define CONCURRENT_NEURAL_NETWORK_H

#include <future>
#include <vector>
#include <map>

#include "neuron.h"
#include "feedback_bus.h"

/**
 * @todo write docs
 */
class concurrent_neural_network {
private:
  unsigned inputs;
  unsigned outputs;

  std::vector<axon*> input_axons;
  std::vector<axon*> output_axons;

  std::vector<neuron*> neurons;
  std::vector<unsigned> concurrent_steps;

  std::map<unsigned, feedback_bus*> feedbackers;

  void add_feedbacker (unsigned origin_neuron, unsigned destiny_neuron, double w);
  void propagate_feedback ();
public:

  concurrent_neural_network(const std::vector< std::vector<bool>>& net_graph,
                            const std::vector< std::vector< double > >& net_costs,
                            const std::vector< unsigned int > concurrent_s,
                            unsigned int inps, unsigned int outs);

  bool operator () (const std::vector<double>& inputs_values,
                    std::vector<double>& outputs_values);
};

#endif // CONCURRENT_NEURAL_NETWORK_H
