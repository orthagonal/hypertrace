#include "report.h"
#include <stdexcept>

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    // Exhaust all truth/HDC/rules combinations; include both classes in each cell.
    struct Case {
        bool truth, hdc, rules;
        unsigned both_correct, both_wrong, hdc_only, rules_only;
        unsigned tp, tn, fp, fn;
    };
    const Case cases[] = {
        {false, false, false, 1,0,0,0, 0,1,0,0},
        {false, false, true,  0,0,1,0, 0,0,1,0},
        {false, true,  false, 0,0,0,1, 0,0,1,0},
        {false, true,  true,  0,1,0,0, 0,0,1,0},
        {true,  false, false, 0,1,0,0, 0,0,0,1},
        {true,  false, true,  0,0,0,1, 1,0,0,0},
        {true,  true,  false, 0,0,1,0, 1,0,0,0},
        {true,  true,  true,  1,0,0,0, 1,0,0,0},
    };
    PredictionComparison total{};
    for (const auto& c : cases) {
        PredictionComparison single{};
        single.add(c.truth, c.hdc, c.rules);
        check(single.both_correct == c.both_correct && single.both_wrong == c.both_wrong &&
              single.hdc_only_correct == c.hdc_only && single.rules_only_correct == c.rules_only,
              "incorrect paired correctness category");
        check(single.either.tp == c.tp && single.either.tn == c.tn &&
              single.either.fp == c.fp && single.either.fn == c.fn,
              "incorrect OR classifier confusion counts");
        total.add(c.truth, c.hdc, c.rules);
    }
    // Asymmetric counts catch swapped categories and ensure repeated rows count.
    total.add(false, true, true);
    for (int i=0; i<2; ++i) total.add(true, true, false);
    for (int i=0; i<3; ++i) total.add(true, false, true);
    check(total.both_correct == 2 && total.both_wrong == 3 &&
          total.hdc_only_correct == 4 && total.rules_only_correct == 5,
          "incorrect accumulated paired counts");
    check(total.either.tp == 8 && total.either.tn == 1 &&
          total.either.fp == 4 && total.either.fn == 1,
          "incorrect accumulated OR confusion counts");
    check(total.both_correct + total.both_wrong + total.hdc_only_correct +
          total.rules_only_correct == 14, "paired counts must cover every record");
    total.print(); // Expected OR precision 0.67, recall 0.89, FP=4.

    PredictionComparison empty{};
    check(empty.both_correct == 0 && empty.both_wrong == 0 &&
          empty.hdc_only_correct == 0 && empty.rules_only_correct == 0 &&
          empty.either.tp == 0 && empty.either.tn == 0 &&
          empty.either.fp == 0 && empty.either.fn == 0, "empty split must have zero counts");
    empty.print(); // Existing Metrics convention: zero precision/recall when undefined.
    std::puts("PASS: paired reporting and OR classifier counts");
}
