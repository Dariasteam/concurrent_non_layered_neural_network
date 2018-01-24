#include <iostream>

#include <vector>
#include <time.h>

std::vector<std::vector<bool>> random_graph_generator() {
  std::vector<std::vector<bool>> vec;
  unsigned size = rand() % 10 + 1;
  vec.resize(size);
  for (unsigned i = 0; i < size; i++) {
    vec[i].resize(size);
    for (unsigned j = 0; j < size; j++) {
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
    for (unsigned j = 0; j < size; j++) {
      if (vec[j][i])
        visited_nodes[i]++;
    }
  }

  for (auto& element : visited_nodes)
    std::cout << element << " ";

  std::cout << "\n\n";

  return visited_nodes;
}

std::vector<std::vector<unsigned>> algorithm (const std::vector<std::vector<bool>>& vec) {
  unsigned size = vec.size();
  std::vector<unsigned> visited_nodes = generate_visited_nodes(vec);

  std::vector<std::vector<unsigned>> solve (size);

  auto aux_visited = visited_nodes;

  for (auto& row : solve)
    row.resize(size);

  unsigned iteration = 1;

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      if (vec[i][j]) {
        if (visited_nodes[i] > 0) {
          iteration++;
          visited_nodes = aux_visited;
          std::cout << "Cambio en el índice " << i << " " << j << std::endl;
        }
        solve[i][j] = iteration;
        std::cout << "Le resto a " << j << " que vale " << visited_nodes[j] << std::endl;
        aux_visited[j]--;
      }
    }
  }
  return solve;
}

int main(int argc, char **argv) {
  srand(time(nullptr));
  auto vec_graph = random_graph_generator();


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


  auto vec_solve = algorithm(vec_graph);

  unsigned size = vec_graph.size();

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      std::cout << vec_graph[i][j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\n\n";

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      std::cout << vec_solve[i][j] << " ";
    }
    std::cout << "\n";
  }


  return 0;
}
