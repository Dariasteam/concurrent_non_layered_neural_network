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
#include <map>
#include <stack>

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
   * @brief Delete neurons with no predecesors (excluding inputs and outputs)
   *
   * @param vec_graph p_vec_graph:...
   * @param vec_costs p_vec_costs:...
   * @param inputs p_inputs:...
   * @param outputs p_outputs:...
   */
  void delete_unreachable_nodes (std::vector<std::vector<bool>>& vec_graph,
                                 std::vector<std::vector<double>>& vec_costs,
                                 unsigned inputs, unsigned outputs);


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
                              unsigned inputs, unsigned outputs);

  /**
   * @brief Extracts the hidden layers of the net and creates a vector of groups
   * of neurons that can be safely calculated concurrently.
   *
   * @param vec p_vec: Cost matriz of the net
   * @return std::vector< unsigned int > groups of neurons conccurent-safe
   */
  std::vector<unsigned> generate_concurrent_steps (const std::vector<std::vector<bool>>& vec);

  /**
   * @brief Generates a vector containing the number of predecesors of each node
   * This function is used by #generate_concurrent_steps to which neurons can
   * be calculated concurrently.
   *
   * @param vec p_vec:...
   * @return std::vector< unsigned int >
   */
  std::vector<unsigned> generate_visited_nodes (const std::vector<std::vector<bool>>& vec);


public:

  concurrent_neural_network(const std::vector<std::vector<bool>>& vec_graph,
                            const std::vector<std::vector<double>>& vec_costs,
                            unsigned int inps, unsigned int outs);

  bool operator () (const std::vector<double>& inputs_values,
                    std::vector<double>& outputs_values);

  unsigned c_steps () { return concurrent_steps.size(); }

};

#endif // CONCURRENT_NEURAL_NETWORK_H
