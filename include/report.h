#pragma once
#include "safety_data.h"
#include <cstdio>
#include <vector>

struct Metrics {
    unsigned tp=0, tn=0, fp=0, fn=0;
    void add(bool truth, bool pred) {
        if (truth && pred) tp++;
        else if (!truth && !pred) tn++;
        else if (!truth && pred) fp++;
        else fn++;
    }
    void print(const char* name) const {
        const double n = tp+tn+fp+fn;
        const double p = tp+fp ? 1.0*tp/(tp+fp) : 0.0;
        const double r = tp+fn ? 1.0*tp/(tp+fn) : 0.0;
        std::printf("%-22s  n=%5.0f  accuracy=%6.2f%%  precision=%5.2f  recall=%5.2f  FP=%u FN=%u\n",
            name, n, n ? 100.0*(tp+tn)/n : 0.0, p, r, fp, fn);
    }
};

// Pair predictions on each record, using the same truth label for both models.
struct PredictionComparison {
    unsigned both_correct=0, both_wrong=0, hdc_only_correct=0, rules_only_correct=0;
    Metrics either{};

    void add(bool truth, bool hdc, bool rules) {
        if (hdc == truth && rules == truth) both_correct++;
        else if (hdc != truth && rules != truth) both_wrong++;
        else if (hdc == truth) hdc_only_correct++;
        else rules_only_correct++;
        either.add(truth, hdc || rules);
    }

    void print() const {
        std::printf("Both models correct:      %u\n", both_correct);
        std::printf("Both models wrong:        %u\n", both_wrong);
        std::printf("HDC correct, rules wrong: %u\n", hdc_only_correct);
        std::printf("Rules correct, HDC wrong: %u\n", rules_only_correct);
        either.print("HDC OR rules");
    }
};

inline const char* split_name(unsigned split) {
    static const char* names[] = {"train", "validation", "test", "long-test"};
    return split <= 3 ? names[split] : "invalid";
}
inline bool basic_baseline(const Record& r) {
    const bool sensitive = (r.flags & (1u<<1)) != 0;
    const bool untrusted = (r.flags & (1u<<2)) != 0;
    const bool approved = (r.flags & (1u<<4)) != 0;
    const bool staged_external = (r.flags & (1u<<5)) != 0;
    return sensitive && untrusted && !approved && staged_external;
}
