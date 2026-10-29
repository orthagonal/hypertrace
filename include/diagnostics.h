#pragma once
#include "report.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace diagnostics {
constexpr unsigned FEATURES = FACT_BITS + RECENT_ACTIONS;
constexpr uint32_t ALL = (1u << FEATURES) - 1;
constexpr uint32_t FACTS = (1u << FACT_BITS) - 1;
using Vector = std::array<uint32_t, HDC_WORDS>;
inline const char* feature_name(unsigned f) {
    static const char* names[] = {
        "seen_private", "payload_sensitive", "inspect_untrusted", "inspect_trusted",
        "approved", "stage_external", "transformed", "unrelated_private",
        "web_search", "requested_approval", "saved", "stage_local",
        "recent_0_newest", "recent_1", "recent_2", "recent_3",
        "recent_4", "recent_5", "recent_6", "recent_7_oldest"
    };
    return f < FEATURES ? names[f] : "invalid";
}
inline unsigned popcount(uint32_t value) {
    unsigned count=0;
    while (value) { value &= value-1; ++count; }
    return count;
}

// Host-only diagnostic encoder. A removed feature contributes NO token, including
// its zero-state token. Recompute the majority denominator; never edit model code.
struct Encoding {
    std::array<Vector, FEATURES> words{};
    std::array<unsigned char, HDC_DIM_BITS> votes{};
    explicit Encoding(const Record& r) {
        for (unsigned f=0; f<FEATURES; ++f)
            for (unsigned w=0; w<HDC_WORDS; ++w) {
                words[f][w] = hyperword(feature_token(r,f),w);
                for (unsigned bit=0; bit<32; ++bit)
                    votes[w*32+bit] += (words[f][w] >> bit) & 1u;
            }
    }
    Vector retained(uint32_t mask=ALL) const {
        if (!mask || (mask & ~ALL)) throw std::invalid_argument("invalid retained feature mask");
        const unsigned count = popcount(mask);
        std::array<unsigned char, HDC_DIM_BITS> remaining = votes;
        for (unsigned f=0; f<FEATURES; ++f) if (!(mask & (1u<<f)))
            for (unsigned w=0; w<HDC_WORDS; ++w)
                for (unsigned bit=0; bit<32; ++bit)
                    remaining[w*32+bit] -= (words[f][w] >> bit) & 1u;
        Vector result{};
        for (unsigned bit=0; bit<HDC_DIM_BITS; ++bit)
            if (2u*remaining[bit] >= count) result[bit/32] |= 1u<<(bit%32);
        return result;
    }
};
struct Distances {
    unsigned safe=0, unsafe=0;
    int margin() const { return int(safe)-int(unsafe); }
    bool prediction() const { return unsafe < safe; } // Ties predict SAFE.
};
inline Distances distances(const Vector& vector, const uint32_t* safe, const uint32_t* unsafe) {
    Distances result;
    for (unsigned w=0; w<HDC_WORDS; ++w) {
        result.safe += popcount(vector[w] ^ safe[w]);
        result.unsafe += popcount(vector[w] ^ unsafe[w]);
    }
    return result;
}
inline const char* length_bucket(uint32_t n) {
    if (n<10) return "0-9";
    if (n<=16) return "10-16";
    if (n<=32) return "17-32";
    if (n<=64) return "33-64";
    if (n<=127) return "65-127";
    if (n<=256) return "128-256";
    if (n<=512) return "257-512";
    return "513+";
}
inline const char* observed_cohort(const Record& r) {
    // Descriptive, disjoint prefix cohorts, NOT reconstructed scenario metadata.
    if (!(r.flags & (1u<<1))) return "observed_public_payload";
    if (r.flags & (1u<<11)) return "observed_local_stage";
    if (r.flags & (1u<<4)) return "observed_approved";
    if (r.flags & (1u<<3)) return "observed_trusted_inspection";
    if ((r.flags & ((1u<<2)|(1u<<5))) == ((1u<<2)|(1u<<5)))
        return "observed_sensitive_untrusted_external_unapproved";
    return "observed_other";
}
inline const char* error_kind(bool truth, bool pred) {
    return truth ? (pred ? "TP" : "FN") : (pred ? "FP" : "TN");
}
inline const char* disagreement(bool truth, bool hdc, bool rules) {
    if (hdc==rules) return "none";
    return hdc==truth ? "hdc_correct_rules_wrong" : "rules_correct_hdc_wrong";
}
inline double accuracy(const Metrics& m) {
    const unsigned n=m.tp+m.tn+m.fp+m.fn;
    return n ? 100.0*(m.tp+m.tn)/n : 0.0;
}
inline Record reorder_recent(Record r, bool reverse) {
    if (reverse) std::reverse(r.recent, r.recent+RECENT_ACTIONS);
    else {
        // Fixed seed, one permutation per record, depends only on observed actions.
        uint32_t seed=42;
        for (auto action : r.recent) seed=mix32(seed ^ action);
        for (unsigned i=RECENT_ACTIONS-1; i>0; --i) {
            seed=mix32(seed+0x9e3779b9u);
            std::swap(r.recent[i], r.recent[seed%(i+1)]);
        }
    }
    return r;
}

struct Options { bool verify=false; std::string directory; };
inline Options options(int argc, char** argv, bool gpu) {
    const std::string usage = gpu
        ? "usage: hdc_safety_gpu <dataset> [--verify] [--diagnostics <directory>]"
        : "usage: hdc_reference <dataset> [--diagnostics <directory>]";
    if (argc<2) throw std::invalid_argument(usage);
    Options result;
    for (int i=2; i<argc; ++i) {
        const std::string arg=argv[i];
        if (arg=="--verify" && gpu && !result.verify) result.verify=true;
        else if (arg=="--diagnostics" && result.directory.empty() && i+1<argc &&
                 std::string(argv[i+1]).find("--")!=0) result.directory=argv[++i];
        else throw std::invalid_argument(usage);
    }
    return result;
}
inline std::ofstream output(const std::filesystem::path& dir, const char* name) {
    std::ofstream out(dir/name);
    if (!out) throw std::runtime_error("cannot write diagnostic file: " + (dir/name).string());
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::fixed << std::setprecision(4);
    return out;
}
struct Group {
    Metrics hdc{}, rules{};
    unsigned n=0, ties=0;
    uint64_t safe=0, unsafe=0;
    int64_t margin=0;
    int min_margin=std::numeric_limits<int>::max(), max_margin=std::numeric_limits<int>::min();
    void add(bool truth, bool rule, const Distances& d) {
        ++n; hdc.add(truth,d.prediction()); rules.add(truth,rule);
        safe+=d.safe; unsafe+=d.unsafe; margin+=d.margin(); ties+=(d.margin()==0);
        min_margin=std::min(min_margin,d.margin()); max_margin=std::max(max_margin,d.margin());
    }
};
struct Intervention {
    Metrics model{};
    unsigned flips=0, fixed=0, broken=0;
    void add(bool truth, bool original, bool changed) {
        model.add(truth,changed); flips+=(original!=changed);
        fixed+=(original!=truth && changed==truth);
        broken+=(original==truth && changed!=truth);
    }
};

inline void run(const std::vector<Record>& data, const uint32_t* safe, const uint32_t* unsafe,
                const std::vector<uint8_t>& predictions, const std::string& directory,
                const std::vector<int16_t>* gpu_margins=nullptr) {
    if (directory.empty()) return;
    if (predictions.size()!=data.size() || (gpu_margins && gpu_margins->size()!=data.size()))
        throw std::invalid_argument("diagnostic prediction/margin count mismatch");
    const std::filesystem::path dir(directory);
    std::filesystem::create_directories(dir);
    auto rows=output(dir,"records.csv"), examples=output(dir,"disagreements.csv");
    const char* header="record_index,split,observed_length,length_bucket,observed_cohort,oracle_label,"
        "oracle_outcome,hdc_prediction,rules_prediction,hdc_error,rules_error,disagreement,"
        "distance_safe,distance_unsafe,margin_safe_minus_unsafe,flags,recent_newest_first,"
        "scenario_type,violation_category,enforcement_outcome\n";
    rows << header; examples << header;
    std::map<std::tuple<unsigned,std::string,std::string>,Group> groups;
    std::array<Group,4> totals{};
    std::array<PredictionComparison,4> paired{};
    constexpr unsigned CONTROLS=4;
    std::array<std::array<Intervention,FEATURES+CONTROLS>,4> interventions{};
    for (size_t i=0; i<data.size(); ++i) {
        const auto& r=data[i];
        if (r.split>=4 || r.label>1) throw std::invalid_argument("invalid diagnostic split/label");
        const Encoding encoded(r);
        const auto vector=encoded.retained();
        // Independent diagnostic implementation must agree with the production encoder.
        for (unsigned w=0; w<HDC_WORDS; ++w)
            if (vector[w]!=encode_word(r,w)) throw std::runtime_error("diagnostic encoder mismatch");
        const auto d=distances(vector,safe,unsafe);
        if (d.prediction()!=(predictions[i]!=0)) throw std::runtime_error("diagnostic prediction mismatch");
        if (gpu_margins && d.margin()!=(*gpu_margins)[i])
            throw std::runtime_error("diagnostic GPU margin mismatch");
        const bool truth=r.label!=0, rule=basic_baseline(r), hdc=d.prediction();
        totals[r.split].add(truth,rule,d); paired[r.split].add(truth,hdc,rule);
        auto add_group=[&](const char* dimension, const char* name) {
            groups[{r.split,dimension,name}].add(truth,rule,d);
        };
        add_group("length",length_bucket(r.prefix_len));
        add_group("observed_cohort",observed_cohort(r));
        add_group("oracle_outcome",truth ? "attempted_violation" : "allowed");
        add_group("hdc_error",error_kind(truth,hdc));
        std::ostringstream row;
        row << i << ',' << split_name(r.split) << ',' << r.prefix_len << ','
            << length_bucket(r.prefix_len) << ',' << observed_cohort(r) << ',' << r.label << ','
            << (truth ? "attempted_violation" : "allowed") << ',' << hdc << ',' << rule << ','
            << error_kind(truth,hdc) << ',' << error_kind(truth,rule) << ','
            << disagreement(truth,hdc,rule) << ',' << d.safe << ',' << d.unsafe << ','
            << d.margin() << ',' << r.flags << ',';
        for (unsigned pos=0; pos<RECENT_ACTIONS; ++pos) {
            if (pos) row << ';';
            row << r.recent[pos];
        }
        row << ",not_recorded,not_recorded,not_recorded\n";
        rows << row.str();
        if (hdc!=rule) examples << row.str();
        for (unsigned f=0; f<FEATURES; ++f) {
            const auto changed=distances(encoded.retained(ALL ^ (1u<<f)),safe,unsafe);
            interventions[r.split][f].add(truth,hdc,changed.prediction());
        }
        for (unsigned c=0; c<CONTROLS; ++c) {
            Vector changed;
            if (c<2) changed=encoded.retained(c==0 ? FACTS : ALL ^ FACTS);
            else changed=Encoding(reorder_recent(r,c==2)).retained();
            interventions[r.split][FEATURES+c].add(truth,hdc,distances(changed,safe,unsafe).prediction());
        }
    }
    auto grouped=output(dir,"groups.csv");
    grouped << "split,dimension,group,n,hdc_accuracy_pct,rules_accuracy_pct,hdc_fp,hdc_fn,rules_fp,rules_fn,"
        "mean_distance_safe,mean_distance_unsafe,mean_margin,min_margin,max_margin,ties\n";
    for (const auto& item : groups) {
        const auto& key=item.first; const auto& g=item.second;
        grouped << split_name(std::get<0>(key)) << ',' << std::get<1>(key) << ',' << std::get<2>(key)
            << ',' << g.n << ',' << accuracy(g.hdc) << ',' << accuracy(g.rules) << ','
            << g.hdc.fp << ',' << g.hdc.fn << ',' << g.rules.fp << ',' << g.rules.fn << ','
            << double(g.safe)/g.n << ',' << double(g.unsafe)/g.n << ',' << double(g.margin)/g.n << ','
            << g.min_margin << ',' << g.max_margin << ',' << g.ties << '\n';
    }
    auto ablations=output(dir,"ablations.csv");
    ablations << "split,intervention,n,baseline_accuracy_pct,accuracy_pct,delta_pp,fp,fn,prediction_flips,"
        "errors_fixed,correct_predictions_broken\n";
    const char* controls[]={"state_only", "recent_only", "reverse_recent", "shuffle_recent_seed42"};
    for (unsigned split=0; split<4; ++split) for (unsigned f=0; f<FEATURES+CONTROLS; ++f) {
        const auto& a=interventions[split][f];
        ablations << split_name(split) << ',' << (f<FEATURES ? std::string("remove_")+feature_name(f) : controls[f-FEATURES])
            << ',' << totals[split].n << ',';
        if (totals[split].n) ablations << accuracy(totals[split].hdc) << ',' << accuracy(a.model) << ','
            << accuracy(a.model)-accuracy(totals[split].hdc);
        else ablations << "NA,NA,NA";
        ablations << ',' << a.model.fp << ',' << a.model.fn << ',' << a.flips << ',' << a.fixed << ',' << a.broken << '\n';
    }
    std::ostringstream text;
    text << std::fixed << std::setprecision(2);
    text << "HDC diagnostics: observed features only; fixed learned prototypes; no retraining.\n"
        "SAFE/UNSAFE denote predicted final oracle class, not enforcement decisions.\n"
        "Margin = distance(SAFE) - distance(UNSAFE); positive predicts UNSAFE; ties predict SAFE.\n";
    for (unsigned split=0; split<4; ++split) {
        const auto& g=totals[split];
        text << '\n' << split_name(split) << " n=" << g.n;
        if (!g.n) { text << " (no records)\n"; continue; }
        text << " HDC=" << accuracy(g.hdc) << "% rules=" << accuracy(g.rules) << "%\n"
            << "  mean distances SAFE=" << double(g.safe)/g.n << " UNSAFE=" << double(g.unsafe)/g.n
            << " margin=" << double(g.margin)/g.n << " range=[" << g.min_margin << ',' << g.max_margin
            << "] ties=" << g.ties << '\n'
            << "  disagreements: HDC correct=" << paired[split].hdc_only_correct
            << " rules correct=" << paired[split].rules_only_correct << '\n';
        for (const auto& item : groups) if (std::get<0>(item.first)==split && std::get<1>(item.first)=="length") {
            const auto& b=item.second;
            text << "  length " << std::get<2>(item.first) << " n=" << b.n << " HDC=" << accuracy(b.hdc)
                << "% rules=" << accuracy(b.rules) << "%\n";
        }
        for (unsigned c=0; c<CONTROLS; ++c) {
            const auto& a=interventions[split][FEATURES+c];
            text << "  " << controls[c] << " accuracy=" << accuracy(a.model)
                << "% delta=" << accuracy(a.model)-accuracy(g.hdc) << " pp flips=" << a.flips << '\n';
        }
        unsigned largest=0;
        for (unsigned f=1; f<FEATURES; ++f)
            if (accuracy(interventions[split][f].model)<accuracy(interventions[split][largest].model)) largest=f;
        text << "  largest single-feature removal loss: " << feature_name(largest) << " delta="
            << accuracy(interventions[split][largest].model)-accuracy(g.hdc) << " pp\n";
    }
    text << "\nLimitations: no stored scenario type, violation subtype, full trajectory, final destination,\n"
        "late-approval marker, or executed enforcement decision. Observed cohorts are descriptive only.\n"
        "Oracle labels score predictions retrospectively; prediction FP/FN are not enforcement failures.\n"
        "No live policy enforcer is exercised. This dataset contains one violation family.\n"
        "Ablations omit tokens at inference with fixed prototypes, change the majority threshold, and\n"
        "can be out of distribution; deltas are model sensitivity, not causal feature importance.\n"
        "Order controls retain action multisets; sensitivity alone does not demonstrate temporal reasoning.\n"
        "Only eight recent positions are encoded. Length is a grouping variable, not an input feature.\n"
        "Train metrics are in-sample; held-out splits share the simulator grammar.\n";
    auto summary=output(dir,"summary.txt"); summary << text.str();
    rows.close(); examples.close(); grouped.close(); ablations.close(); summary.close();
    std::printf("\n%s\nDiagnostic artifacts: %s\n",text.str().c_str(),directory.c_str());
}
} // namespace diagnostics
