#include<iostream>
#include<vector>
#include<set>

std::set<unsigned int> states;
bool done = false;

bool finished_state(unsigned int state) {
    return state == 0 || state == (1 << 20) - 1;
}

unsigned int change(unsigned int state, int i, int j) {
    if (i < 0 || i > 3) return state;
    if (j < 0 || j > 4) return state;
    int bit = i * 5 + j;
    state ^= (1 << bit);
    return state;
}

unsigned int generate_state(unsigned int state, int index) {
    int i = index / 5;
    int j = index % 5;

    state = change(state, i - 1, j);
    state = change(state, i + 1, j);
    state = change(state, i, j - 1);
    state = change(state, i, j + 1);
    return state;
}

void recursive(unsigned int state, std::vector<int>& path) {
    if (states.count(state) > 0) return;
    if (done) return;
    if (path.size() > 100) return;
    states.insert(state);
    std::cout << "Checking state " << state << ", path " << path.size() << std::endl;
    if (finished_state(state)) {
        std::cout << "Path:\n";
        for (const auto& step : path) {
            std::cout << step << " ";
        }
        std::cout << std::endl;
        done = true;
        return;
    }

    for (int i = 0; i < 20; i++) {
        path.push_back(i);
        unsigned int new_state = generate_state(state, i);
        recursive(new_state, path);
        path.pop_back();
    }

    return;
}

int main(void) {
    unsigned int state = 64;
    std::vector<int> path;

    recursive(state, path);
    return 0;
}
