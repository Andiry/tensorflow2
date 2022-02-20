#include<iostream>
#include<vector>

void flip(std::vector<std::vector<int>>& matrix, int i, int j) {
	std::vector<std::vector<int>> dir = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
	int m = matrix.size();
	int n = matrix[0].size();
	for (const auto& d : dir) {
		int k = i + d[0];
		int l = j + d[1];
		if (k >= 0 && k < m && l >= 0 && l < n) {
			matrix[k][l] = 1 - matrix[k][l];
		}
	}
}

bool finish(const std::vector<std::vector<int>>& matrix) {
	int m = matrix.size();
	int n = matrix[0].size();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (matrix[i][j] != matrix[0][0]) return false;
		}
	}
	return true;
}

bool resolve(std::vector<std::vector<int>>& matrix,
		std::vector<std::pair<int, int>>& path,
		int i, int j) {
	if (finish(matrix))
		return true;

	int m = matrix.size();
	int n = matrix[0].size();

	if (i >= m || j >= n)
		return false;

	int i_next = j == n - 1 ? i + 1 : i;
	int j_next = j == n - 1 ? 0 : j + 1;

	if (resolve(matrix, path, i_next, j_next))
		return true;

	flip(matrix, i, j);
	path.push_back({i, j});
	if (resolve(matrix, path, i_next, j_next))
		return true;

	flip(matrix, i, j);
	path.pop_back();
	return false;
}

void print_path(const std::vector<std::pair<int, int>>& path) {
	for (const auto& index : path) {
		std::cout << index.first << ", " << index.second << std::endl;
	}
}

int main(void) {
	std::vector<std::vector<int>> matrix = {{0, 1, 0, 0, 0},
						{1, 0, 0, 0, 0},
						{1, 0, 1, 0, 1},
						{0, 0, 0, 0, 1},
						{0, 0, 0, 1, 0}};
	std::vector<std::pair<int, int>> path;
	bool res = resolve(matrix, path, 0, 0);
	if (res)
		print_path(path);
	else
		std::cout << "No answer." << std::endl;
	return 0;
}
